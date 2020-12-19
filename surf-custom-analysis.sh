#!/bin/bash                                                                     
#SBATCH -t 24:00:00                                                             
#SBATCH -N 1                                                                    
#SBATCH --mem=60G                                                               
#SBATCH --mail-type=END                                                         
#SBATCH --mail-user=m.b.cetin@vu.nl                                             
#SBATCH -e slurm-%j.err                                                         
#SBATCH -o slurm-%j.out                                                         

/home/cmt2002/surfsara-tool/analyze.sh -m node_load1 -r r23
/home/cmt2002/surfsara-tool/analyze.sh -m surfsara_power_usage -r r23
/home/cmt2002/surfsara-tool/analyze.sh -m surfsara_ambient_temp -r r23
/home/cmt2002/surfsara-tool/analyze.sh -m node_memory_MemFree,node_memory_MemTotal -r r23
