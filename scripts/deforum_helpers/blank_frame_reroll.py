from .generate import generate
#WebUI
from modules.shared import opts, cmd_opts, state

def blank_frame_reroll(image, args, root, frame_idx):
    patience = 10
    print("Blank frame detected! If you don't have the NSFW filter enabled, this may be due to a glitch!")
    if args.reroll_blank_frames == 'reroll':
        while not image.getbbox():
            print("Rerolling with +1 seed...")
            args.seed += 1
            image = generate(args, root, frame_idx)
            patience -= 1
            if patience == 0:
                print("Rerolling with +1 seed failed for 10 iterations! Try setting webui's precision to 'full' and if it fails, please report this to the devs! Interrupting...")
                state.interrupted = True
                state.current_image = image
                return None
    elif args.reroll_blank_frames == 'interrupt':
        print("Interrupting to save your eyes...")
        state.interrupted = True
        state.current_image = image
        return None
    return image