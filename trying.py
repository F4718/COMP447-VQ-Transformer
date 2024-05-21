from model import VQ3DDecoder, VQ3DEncoder, CodeBook
import torch


enc = VQ3DEncoder(1, 256, 2, 4, 2, True)
dec = VQ3DDecoder(256, 1, 2, 4, 2, True)
quant = CodeBook(128, 256)
inp = torch.randn(16, 1, 2, 128, 128)

with torch.no_grad():
    print(inp.shape)
    h = enc(inp)
    print(h.shape)
    q, _, i = quant(h)
    print(q.shape)
    print(i.shape)
    rec = dec(q)
    print(rec.shape)