from model import VQ3DDecoder, VQ3DEncoder, CodeBook
import torch


enc = VQ3DEncoder(1, 256, 4, 4, 2, True)
dec = VQ3DDecoder(256, 1, 4, 4, 2, True)
quant = CodeBook(128, 256)
inp = torch.randn(16, 1, 4, 128, 128)

with torch.no_grad():
    print(inp.shape)
    h = enc(inp)
    print(h.shape)
    q, _, i = quant(h)
    print(q.shape)
    print(i.shape)
    rec = dec(q)
    print(rec.shape)

    r = torch.arange(0, 10)
    rr = torch.randn(10, 16, 8)
    print(rr)
    print(rr[1::2])
    print(rr[1::2].shape)
    print(rr[1::2][:-1].shape)
    print(r.view(5, 2))

    x = torch.randn(5,6,7,8,9)
    print(x[:,0].shape)


