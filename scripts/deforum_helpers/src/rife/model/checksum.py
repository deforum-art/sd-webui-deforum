import os
import hashlib

def checksum(filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128):
    h = hash_factory()
    with open(filename,'rb') as f: 
        while chunk := f.read(chunk_num_blocks*h.block_size): 
            h.update(chunk)
    return h.hexdigest()


#print(checksum('D:/D-SD/autopt2NEW/stable-diffusion-webui/models/Deforum/RIFE46.pkl'))