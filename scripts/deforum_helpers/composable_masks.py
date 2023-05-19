# At the moment there are three types of masks: mask from variable, file mask and word mask
# Variable masks include video_mask (which can be set to auto-generated human masks) and everywhere
# They are put in {}-brackets
# Word masks are framed with <>-bracets, like: <cat>, <anime girl>
# File masks are put in []-brackes
# Empty strings are counted as the whole frame
# We want to put them all into a sequence of boolean operations

# Example:
# \ <armor>
# (({human_mask} & [mask1.png]) ^ <apple>)

# Writing the parser for the boolean sequence
# using regex and PIL operations
import re
from .load_images import get_mask_from_file, check_mask_for_errors, blank_if_none
from .word_masking import get_word_mask
from PIL import ImageChops
from modules.shared import opts

# val_masks: name, PIL Image mask
# Returns an image in mode '1' (needed for bool ops), convert to 'L' in the sender function
def compose_mask(root, args, mask_seq, val_masks, frame_image, inner_idx:int = 0):
    # Compose_mask recursively: go to inner brackets, then b-op it and go upstack
    
    # Step 1:
    # recursive parenthesis pass
    # regex is not powerful here

    seq = ""
    inner_seq = ""
    parentheses_counter = 0

    for c in mask_seq:
        if c == ')':
            parentheses_counter = parentheses_counter - 1
        if parentheses_counter > 0:
            inner_seq += c
        if c == '(':
            parentheses_counter = parentheses_counter + 1
        if parentheses_counter == 0:
            if len(inner_seq) > 0:
                inner_idx += 1
                seq += compose_mask(root, args, inner_seq, val_masks, frame_image, inner_idx)
                inner_seq = ""
            else:
                seq += c
    
    if parentheses_counter != 0:
        raise Exception('Mismatched parentheses in {mask_seq}!')

    mask_seq = seq

    # Step 2:
    # Load the word masks and file masks as vars
    
    # File masks
    pattern = r'\[(?P<inner>[\S\s]*?)\]'
    
    def parse(match_object):
        nonlocal inner_idx
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[str(inner_idx)] = get_mask_from_file(content, args).convert('1') # TODO: add caching
        return f"{{{inner_idx}}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Word masks
    pattern = r'<(?P<inner>[\S\s]*?)>'
    
    def parse(match_object):
        nonlocal inner_idx
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[str(inner_idx)] = get_word_mask(root, frame_image, content).convert('1')
        return f"{{{inner_idx}}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Now that all inner parenthesis are eliminated we're left with a linear string
    
    # Step 3:
    # Boolean operations with masks
    # Operators: invert !, and &, or |, xor ^, difference \
    
    # Invert vars with '!'
    pattern = r'![\S\s]*{(?P<inner>[\S\s]*?)}'
    def parse(match_object):
        nonlocal inner_idx
        inner_idx += 1
        content = match_object.groupdict()['inner']
        savename = content
        if content in root.mask_preset_names:
            inner_idx += 1
            savename = str(inner_idx)
        val_masks[savename] = ImageChops.invert(val_masks[content])
        return f"{{{savename}}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Multiply neighbouring vars with '&'
    # Wait for replacements stall (like in Markov chains)
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*&[\s]*{(?P<inner2>[\S\s]*?)}'
        def parse(match_object):
            nonlocal inner_idx
            inner_idx += 1
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_and(val_masks[content], val_masks[content_second])
            return f"{{{savename}}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break
    
    # Add neighbouring vars with '|'
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*?\|[\s]*?{(?P<inner2>[\S\s]*?)}'
        def parse(match_object):
            nonlocal inner_idx
            inner_idx += 1
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_or(val_masks[content], val_masks[content_second])
            return f"{{{savename}}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break
    
    # Mutually exclude neighbouring vars with '^'
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*\^[\s]*{(?P<inner2>[\S\s]*?)}'
        def parse(match_object):
            nonlocal inner_idx
            inner_idx += 1
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_xor(val_masks[content], val_masks[content_second])
            return f"{{{savename}}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break
    
    # Set-difference the regions with '\'
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*\\[\s]*{(?P<inner2>[\S\s]*?)}'
        def parse(match_object):
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                nonlocal inner_idx
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_and(val_masks[content], ImageChops.invert(val_masks[content_second]))
            return f"{{{savename}}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break
    
    # Step 4:
    # Output
    # Now we should have a single var left to return. If not, raise an error message
    pattern = r'{(?P<inner>[\S\s]*?)}'
    matches = re.findall(pattern, mask_seq)
    
    if len(matches) != 1:
        raise Exception(f'Wrong composable mask expression format! Broken mask sequence: {mask_seq}')
    
    return f"{{{matches[0]}}}"

def compose_mask_with_check(root, args, mask_seq, val_masks, frame_image):
    for k, v in val_masks.items():
        val_masks[k] = blank_if_none(v, args.W, args.H, '1').convert('1')
    return check_mask_for_errors(val_masks[compose_mask(root, args, mask_seq, val_masks, frame_image, 0)[1:-1]].convert('L'))
