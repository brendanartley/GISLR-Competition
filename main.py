from gislr_lightning.training import train

def main(args):
    module = train(
        max_epochs=args.max_epochs,
        num_workers=2,
        project='my_awesome_project',
        # project='gislr',
        fast_dev_run=True,
    )
    pass

if __name__ == "__main__":
    main()