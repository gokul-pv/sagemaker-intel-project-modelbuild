import os
import argparse
import subprocess

from pathlib import Path
from git.repo.base import Repo
from smexperiments.tracker import Tracker
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from utils import extract_archive

dvc_repo_url = os.environ.get('DVC_REPO_URL')
dvc_branch = os.environ.get('DVC_BRANCH')

git_user = os.environ.get('GIT_USER', "sagemaker")
git_email = os.environ.get('GIT_EMAIL', "sagemaker-processing@example.com")

ml_root = Path("/opt/ml/processing")

dataset_zip = ml_root / "input" / "intel.zip"
git_path = ml_root / "sagemaker-intel"

def configure_git():
    subprocess.check_call(['git', 'config', '--global', 'user.email', f'"{git_email}"'])
    subprocess.check_call(['git', 'config', '--global', 'user.name', f'"{git_user}"'])
    
def clone_dvc_git_repo():
    print(f"\t:: Cloning repo: {dvc_repo_url}")
    
    repo = Repo.clone_from(dvc_repo_url, git_path.absolute(), allow_unsafe_protocols=True)
    
    return repo

def sync_data_with_dvc(repo):
    os.chdir(git_path)
    print(f":: Create branch {dvc_branch}")
    try:
        repo.git.checkout('-b', dvc_branch)
        print(f"\t:: Create a new branch: {dvc_branch}")
    except:
        repo.git.checkout(dvc_branch)
        print(f"\t:: Checkout existing branch: {dvc_branch}")
    print(":: Add files to DVC")
    
    subprocess.check_call(['dvc', 'add', "dataset"])
    
    repo.git.add(all=True)
    repo.git.commit('-m', f"'add data for {dvc_branch}'")
    
    print("\t:: Push data to DVC")
    subprocess.check_call(['dvc', 'push'])
    
    print("\t:: Push dvc metadata to git")
    repo.remote(name='origin')
    repo.git.push('--set-upstream', repo.remote().name, dvc_branch, '--force')

    sha = repo.head.commit.hexsha
    
    print(f":: Commit Hash: {sha}")
    
    # with Tracker.load() as tracker:
    #     tracker.log_parameters({"data_commit_hash": sha})

def write_dataset(image_paths, output_dir):
    for (data, _), (img_path, _) in zip(image_paths, image_paths.imgs):     
        Path(output_dir / Path(img_path).parent.stem).mkdir(parents=True, exist_ok=True)
        save_image(data, output_dir / Path(img_path).parent.stem / Path(img_path).name)

def generate_train_test_split():
    dataset_extracted = ml_root / "tmp"
    dataset_extracted.mkdir(parents=True, exist_ok=True)
    
    # split dataset and save to their directories
    print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
    extract_archive(
        from_path=dataset_zip,
        to_path=dataset_extracted
    )
    transforms = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    trainset = ImageFolder(dataset_extracted / "seg_train" / "seg_train", transform=transforms)
    testset = ImageFolder(dataset_extracted / "seg_test" / "seg_test", transform=transforms)
    
    for path in ['train', 'test']:
        output_dir = git_path / "dataset" / path
        print(f"\t:: Creating Directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(":: Saving Datasets")
    write_dataset(trainset, git_path / "dataset" / "train")
    write_dataset(testset, git_path / "dataset" / "test")
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # setup git
    print(":: Configuring Git")
    configure_git()
    
    print(":: Cloning Git")
    repo = clone_dvc_git_repo()
    
    print(":: Generate Train Test Split")
    # extract the input zip file and split into train and test
    generate_train_test_split()
        
    print(":: copy data to train")
    subprocess.check_call('cp -r /opt/ml/processing/sagemaker-intel/dataset/train/* /opt/ml/processing/dataset/train', shell=True)
    subprocess.check_call('cp -r /opt/ml/processing/sagemaker-intel/dataset/test/* /opt/ml/processing/dataset/test', shell=True)
    
    print(":: Sync Processed Data to Git & DVC")
    sync_data_with_dvc(repo)

 