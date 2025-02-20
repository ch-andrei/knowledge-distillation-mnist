from trainer import *


def test():
    # set seed for reproducibility
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    train_loader_jitter, train_loader, test_loader = get_dataloaders()

    output_dir = None  # do not save models

    dim_hidden = 800
    use_dropout = False
    regularize_weights = False
    use_jitter = False

    model_t = get_model(dim_hidden=dim_hidden, num_hidden=2, device=device, use_dropout=use_dropout)

    logger("dim_hidden", dim_hidden, 'use_dropout', use_dropout, 'regularize_weights', regularize_weights)
    logger(f"Training teacher network on {'train_loader_jitter' if use_jitter else 'train_loader'}...")

    train_model(
        model_t, device, train_loader_jitter if use_jitter else train_loader, test_loader, output_dir,
        output_tag="",
        regularize_weights=regularize_weights
    )
    test_model(model_t, test_loader, device, verbose=True)


if __name__ == "__main__":
    test()
