### Fork of [PhysBench](https://github.com/physical-superintelligence-lab/PhysBench.git)

<!-- Customized evaluation code in `eval_zp`. -->

~~~shell
# git clone repo
# git clone https://github.com:ZpWang-AI/PhysBench.git
git clone https://github.com/physical-superintelligence-lab/PhysBench.git
# download dataset
cd PhysBench/eval/physbench
huggingface-cli download USC-GVL/PhysBench --local-dir . --local-dir-use-symlinks False --repo-type dataset
# unzip files
yes | unzip image.zip -d image
yes | unzip video.zip -d video
~~~