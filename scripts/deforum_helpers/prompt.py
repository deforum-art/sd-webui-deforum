import re

def split_weighted_subprompts(text, frame = 0):
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
        print(f'Positive prompt:{positive_prompts}')
        print(f'Negative prompt:{negative_prompts}')
    else:
        positive_prompts = prompt_split[0]
        print(f'Positive prompt:{positive_prompts}')
        negative_prompts = ""

    return positive_prompts, negative_prompts
