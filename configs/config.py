CFG = {
    "data": {
        "images_path": "data/TWCC/IMG/*.png",
        "masks_path": "data/TWCC/MASK/*.png",
        "test_images_path": "data/test/IMG/*.png",
        "test_masks_path": "data/test/MASK/*.png",
        "image_size": 256,
        "load_with_info": True
    },
    "train": {
        "batch_size": 8,
        "buffer_size": 1000,
        "epoches": 1,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [256, 256, 3],  # modify this to our model
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}
