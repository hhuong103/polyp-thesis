"""Compatibility entry point; all training is implemented in src.train."""
try:
    from .train import main
except ImportError:
    from train import main

if __name__ == '__main__':
    main()
