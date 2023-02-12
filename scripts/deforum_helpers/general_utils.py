import hashlib

# TODO: move this to a util file and remove the duplication of it in rife's src model checksum.py
def checksum(filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128):
    
    h = hash_factory()
    with open(filename,'rb') as f: 
        while chunk := f.read(chunk_num_blocks*h.block_size): 
            h.update(chunk)
    return h.hexdigest()
    
def get_os():
    import platform
    os_name = platform.system()
    if os_name == "Windows":
        return "Windows"
    elif os_name == "Linux":
        return "Linux"
    elif os_name == "Darwin":
        return "Mac"
    else:
        return "Unknown"