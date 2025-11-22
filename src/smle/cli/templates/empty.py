from smle import SMLE

app = SMLE()

@app.entrypoint
def main(args):
    # 'args' contains your smle.yaml configurations
    print(f"Training with learning rate: {args['training']['lr']}")

    # Your logic here...

if __name__ == "__main__":
    app.run()