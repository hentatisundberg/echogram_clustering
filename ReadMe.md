
# Training models for echogram segmentation 

# Run examples

# Select images
python code/select_diverse_images.py \
    --input "data/all" \
    --output "data/select1" \
    --n 20 \
    --method fps


# Active-learning loop (semantic segmentation)

Everything runs on the remote machine. The annotation GUI is a small web page served by
`code/al.py serve`. It uses only the Python standard library, so there is nothing extra to install.
You open it in your laptop's browser through an SSH tunnel. No X11 or display is needed.

Masks are stored as uint8 PNGs with class ids `0 near field, 1 seafloor, 2 surface turbidity,
3 background` (the ids the current model uses), and `255 = ignore`. Paint unsure regions as
ignore: training will skip them.

```bash
# 0. On the remote, work inside tmux so jobs survive SSH disconnects
tmux new -s al

# 1. Create the workspace (data/al), import the existing Roboflow labels as masks
python code/al.py init --model runs/semantic/train/weights/best.pt \
    --pool data/all --roboflow roboflow_download/echograms-2-3

# 2. Score unlabelled images, queue the 20 most informative
#    (--strategy mixed = 75% most uncertain + 25% random; --sample 2000 to score a subset)
python code/al.py select --n 20

# 3. Annotate. Remote:
python code/al.py serve
#    Laptop (separate terminal):
ssh -N -L 8765:localhost:8765 user@remote
#    then open http://localhost:8765   (press ? in the page for shortcuts)

# 4. Fine-tune the current model on everything labelled so far
python code/al.py train --epochs 60 --device 0,1 --repredict

# 5. Repeat from step 2.  Check progress with:
python code/al.py status
```

Annotation tips: "Fill below" (`D`) fills everything under a polyline you click (good for seafloor), and
"Fill above" (`U`) does the same upwards (good for surface turbidity). "Only paint over" protects other classes while you paint.
The "Low conf" toggle (`C`) highlights where the model was unsure.
