import os
import random
import shutil

# Klasör yolları
train_dir = 'data/chest_xray/train'
val_dir = 'data/chest_xray/val'
classes = ['NORMAL', 'PNEUMONIA']

for cls in classes:
    # Train klasöründeki resimleri listele
    files = os.listdir(os.path.join(train_dir, cls))
    # 100 tanesini rastgele seç
    move_files = random.sample(files, 100)

    for f in move_files:
        src = os.path.join(train_dir, cls, f)
        dst = os.path.join(val_dir, cls, f)
        shutil.move(src, dst)

print("Veriler taşındı, artık sağlam bir val setin var brom!")