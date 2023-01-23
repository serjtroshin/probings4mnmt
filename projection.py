import torch

def stretch_along(emb, direction, gamma=1.0):
    # emb: (N, d)
    # direction: (M, d)
    # gamma: float, torch.tensor
    # returns: (N, M, d) matrix, emb + gamma * dot(emb, direction) * direction
    direction = direction / torch.norm(direction, dim=-1, keepdim=True)
    return  emb.unsqueeze(1) + gamma * torch.sum(emb[:, None, :] * direction[None, :, :], dim=-1, keepdims=True) * direction[None, :, :]

def main():
    emb = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float)
    direction = torch.tensor([[1, 1, 1]], dtype=torch.float)
    print(stretch_along(emb, direction))
    print(stretch_along(emb, direction, gamma=2.0))
    direction = torch.tensor([[1, 0.0, 0.0], [0.0, 1, 0.0]], dtype=torch.float)
    print(stretch_along(emb, direction, gamma=1.0))

if __name__ == '__main__':
    main()
