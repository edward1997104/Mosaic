from dataclasses import dataclass
import tyro
import os
import glob
import numpy as np
import mcubes
import multiprocessing

@dataclass
class Args:
    input_folder : str
    output_folder : str
    workers : int = 8

args = tyro.cli(Args)

def process_one(file_to_convert):
    obj_save_path = os.path.join(args.output_folder, os.path.basename(file_to_convert).split(".")[0] + ".obj")
    npz = np.load(file_to_convert)
    voxel_grid = npz["voxel_grid"]
    voxel_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
    vertices, triangles = mcubes.marching_cubes(voxel_grid, 0.5)
    mcubes.export_obj(vertices, triangles, obj_save_path)
    print("Saved to: ", obj_save_path)

def worker(queue, count):

    while True:
        item = queue.get()
        if item is None:
            break
        try:
            process_one(item)
        except Exception as e:
            print(e)
        queue.task_done()
        with count.get_lock():
            count.value += 1

if __name__ == '__main__':
    files_to_convert = glob.glob(os.path.join(args.input_folder, "*.npz"))

    print("Number of files to convert: ", len(files_to_convert))

    os.makedirs(args.output_folder, exist_ok=True)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    for worker_i in range(args.workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count)
        )
        # process.daemon = True
        process.start()

    for file in files_to_convert:
        queue.put(file)

    queue.join()

    for _ in range(args.workers):
        queue.put(None)

    print("Done")




