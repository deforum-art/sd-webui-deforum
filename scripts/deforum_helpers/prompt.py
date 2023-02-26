import re

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

def parse_weight(match, frame = 0, max_frames = 0)->float:
    import numexpr
    w_raw = match.group("weight")
    max_f = max_frames
    if w_raw == None:
        return 1
    if check_is_number(w_raw):
        return float(w_raw)
    else:
        t = frame
        if len(w_raw) < 3:
            print('the value inside `-characters cannot represent a math function')
            return 1
        return float(numexpr.evaluate(w_raw[1:-1]))

def split_weighted_subprompts(text, frame = 0, max_frames = 0):
    """
    splits the prompt based on deforum webui implementation, moved from generate.py 
    """
    math_parser = re.compile("""
            (?P<weight>(
            `[\S\s]*?`# a math function wrapped in `-characters
            ))
            """, re.VERBOSE)
    
    parsed_prompt = re.sub(math_parser, lambda m: str(parse_weight(m, frame)), text)

    negative_prompts = []
    positive_prompts = []

    prompt_split = parsed_prompt.split("--neg")
    if len(prompt_split) > 1:
        positive_prompts, negative_prompts = parsed_prompt.split("--neg") #TODO: add --neg to vanilla Deforum for compat
    else:
        positive_prompts = prompt_split[0]
        negative_prompts = ""

    return positive_prompts, negative_prompts

def interpolate_prompts(animation_prompts, max_frames):
    import numpy as np
    import pandas as pd
    # Get prompts sorted by keyframe 
    sorted_prompts = sorted(animation_prompts.items(), key=lambda item: int(item[0]))

    # Setup container for interpolated prompts
    prompt_series = pd.Series([np.nan for a in range(max_frames)])

    # For every keyframe prompt except the last
    for i in range(0,len(sorted_prompts)-1):
        
        # Get current and next keyframe
        current_frame = int(sorted_prompts[i][0])
        next_frame = int(sorted_prompts[i+1][0])
        
        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_frame>=next_frame:
            print(f"WARNING: Sequential prompt keyframes {i}:{current_frame} and {i+1}:{next_frame} are not monotonously increasing; skipping interpolation.")
            continue
            
        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i+1][1]
        current_positive, current_negative, *_ = current_prompt.split("--neg") + [None]
        next_positive, next_negative, *_ = next_prompt.split("--neg") + [None]
        
        # Calculate how much to shift the weight from current to next prompt at each frame
        weight_step = 1/(next_frame-current_frame)
        
        # Apply weighted prompt interpolation for each frame between current and next keyframe
        # using the syntax:  prompt1 :weight1 AND prompt1 :weight2 --neg nprompt1 :weight1 AND nprompt1 :weight2
        # (See: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion )
        for f in range(current_frame,next_frame):
            next_weight = weight_step * (f-current_frame)
            current_weight = 1 - next_weight
            
            # We will build the prompt incrementally depending on which prompts are present
            prompt_series[f] = ''

            # Cater for the case where neither, either or both current & next have positive prompts:
            if current_positive:
                prompt_series[f] += f"{current_positive} :{current_weight}"
            if current_positive and next_positive:
                prompt_series[f] += f" AND "
            if next_positive:
                prompt_series[f] += f"{next_positive} :{next_weight}"
            
            # Cater for the case where neither, either or both current & next have negative prompts:
            if current_negative or next_negative:
                prompt_series[f] += " --neg "
                if current_negative:
                    prompt_series[f] += f" {current_negative} :{current_weight}"
                if current_negative and next_negative:
                    prompt_series[f] += f" AND "
                if next_negative:
                    prompt_series[f] += f" {next_negative} :{next_weight}"
    
    # Set explicitly declared keyframe prompts (overwriting interpolated values at the keyframe idx). This ensures:
    # - That final prompt is set, and
    # - Gives us a chance to emit warnings if any keyframe prompts are already using composable diffusion
    for i, prompt in animation_prompts.items():
        prompt_series[int(i)] = prompt
        if ' AND ' in prompt:
            print(f"WARNING: keyframe {i}'s prompt is using composable diffusion (aka the 'AND' keyword). This will cause unexpected behaviour with interpolation.")
    
    # Return the filled series, in case max_frames is greater than the last keyframe or any ranges were skipped.
    return prompt_series.ffill().bfill()
