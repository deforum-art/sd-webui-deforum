# At the moment there are three types of masks: mask from variable, file mask and word mask
# Variable masks include init_mask for the predefined whole-video mask, frame_mask from video-masking system
# and human_mask for a model which better segments people in the background video
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
from torch import Tensor
import PIL
from PIL import Image, ImageChops

# val_masks: name, PIL Image mask
# Returns an image in mode '1' (needed for bool ops), convert to 'L' in the sender function
def compose_mask(args, mask_seq:str, val_masks:dict[str, PIL.Image], frame_image:Tensor, inner_idx:int = 0) -> Image:
    # Compose_mask recursively: go to inner brackets, then b-op it and go upstack
    # pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\)([,][\s]?[\s]?$))'
    
    # Step 1:
    # recursive parenthesis pass
    pattern = r'[\(](?P<inner>)[\)]'
    
    def parse(match_object):
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[f"{{inner_idx}}"] = compose_mask(content, val_masks, frame_image, inner_idx)
        return f"{{inner_idx}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Step 2:
    # Load the word masks and file masks as vars
    
    # File masks
    pattern = r'\[(?P<inner>[\S\s]*?)\]'
    
    def parse(match_object):
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[f"{{inner_idx}}"] = get_mask_from_file(content, args).convert('1') # TODO: add caching
        return f"{{inner_idx}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Word masks
    pattern = r'\<(?P<inner>[\S\s]*?)\>'
    
    def parse(match_object):
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[f"{{inner_idx}}"] = get_word_mask(root, frame_image, content).convert('1')
        return f"{{inner_idx}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Now that all inner parenthesis are eliminated we're left with a linear string
    
    # Step 3:
    # Boolean operations with masks
    # Operators: invert !, and &, or |, xor ^, difference \
    
    # Invert vars with '!'
    pattern = r'![\S\s]*\{(?P<inner>[\S\s]*?)\}'
    def parse(match_object):
        content = match_object.groupdict()['inner']
        val_masks[content] = ImageChops.invert(val_masks[content])
        return f"{{content}}"
    
    mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Multiply neighbouring vars with '&'
    # Wait for replacements stall (like in Markov chains)
    prev_mask_seq = mask_seq
    while mask_seq is not prev_mask_seq:
        pattern = r'\{(?P<inner1>?)\}[\S\s]*&[\S\s]*\{(?P<inner2>?)\}'
        def parse(match_object):
            content = match_object.groupdict()['inner']
            content_second = match_object.groupdict()['inner2']
            val_masks[content] = ImageChops.logical_and(val_masks[content], val_masks[content_second])
            return f"{{content}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Add neighbouring vars with '|'
    prev_mask_seq = mask_seq
    while mask_seq is not prev_mask_seq:
        pattern = r'\{(?P<inner1>?)\}[\S\s]*\|[\S\s]*\{(?P<inner2>?)\}'
        def parse(match_object):
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            val_masks[content] = ImageChops.logical_or(val_masks[content], val_masks[content_second])
            return f"{{content}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Mutually exclude neighbouring vars with '^'
    prev_mask_seq = mask_seq
    while mask_seq is not prev_mask_seq:
        pattern = r'\{(?P<inner1>?)\}[\S\s]*\^[\S\s]*\{(?P<inner2>?)\}'
        def parse(match_object):
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            val_masks[content] = ImageChops.logical_xor(val_masks[content], val_masks[content_second])
            return f"{{content}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Set-difference the regions with '\'
    prev_mask_seq = mask_seq
    while mask_seq is not prev_mask_seq:
        pattern = r'\{(?P<inner1>?)\}[\S\s]*\\[\S\s]*\{(?P<inner2>?)\}'
        def parse(match_object):
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            val_masks[content] = ImageChops.logical_and(val_masks[content], ImageChops.invert(val_masks[content_second]))
            return f"{{content}}"
        
        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
    
    # Step 4:
    # Output
    # Now we should have a single var left to return. If not, raise an error message
    pattern = r'\{(?P<inner>?)\}'
    matches = re.findall(pattern, mask_seq)
    
    if len(matches) != 1:
        raise Exception(f'Wrong composable mask expression format! Broken mask sequence: {mask_seq}')
    
    return matches[0].groupdict()['inner']

def compose_mask_with_check(args, mask_seq:str, val_masks:dict[str, PIL.Image], frame_image:Tensor) -> Image:
    for k, v in val_masks:
        val_masks[k] = blank_if_none(v, args.W, args.H, '1').convert('1')
    return check_mask_for_errors(compose_mask(mask_seq, val_masks, frame_image, 0).convert('L'))
