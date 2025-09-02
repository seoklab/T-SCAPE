# T-SCAPE: T-cell Immunogenicity Scoring via Cross-domain Aided Predictive Engine
This repository represents a novel T-cell epitope immunogenicity prediction model that leverages a multi-domain deep learning approach. Technical details and thorough analysis can be found in our paper, [T-SCAPE: T-cell Immunogenicity Scoring via Cross-domain Aided Predictive Engine](https://www.biorxiv.org/content/10.1101/2025.05.11.653308v1), written by Jeonghyeon Kim, Nuri Jung, Jayyoon Lee, Nam-Hyuk Cho, Jinsung Noh, and Chaok Seok. If you have any question, feel free to open an issue or reach out at jeonghyeonkim86@gmail.com.

## Installation guide(linux only)
1. Install [Anaconda](https://www.anaconda.com/download) if you have not installed it yet.
2. Installation can be done by running below commands in terminal from main directory location. After git clone, below commands should be run in terminal from main directory location.
3. Clone this repository
```
$ git clone https://github.com/seoklab/TITANiAN.git
```
4. Create a conda environment using following commands
```
conda create -n immuno python=3.10
conda activate immuno
conda install numpy matplotlib scikit-learn pandas wandb
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
5. Download the parameter from [google drive](https://drive.google.com/drive/folders/12EqwOtTEX7VCgP3LQIRegpaK1JJiAjjd?usp=sharing) and place it in the main directory location.

Total installation would take ~10min for normal computer.

## Usage guide (linux only)
Below commands should be run in terminal from main directory location. Average runtime is ~5min. 

### p-mhc immunogenicity prediction for cancer immunotherapy development

For this category, the CSV file must include Allele and peptide columns.

Allele represents allele type of each MHC molecule. The allele types should be selected from the allele column of MHC_classI_pseudo.csv. While it is possible to use other allele names, the accuracy of the results cannot be guaranteed.

peptide represents peptide you want to check immunogenicity. Each character in the peptide column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports peptides up to a maximum length of 20 amino acids, with optimal performance observed with 9-mer peptides.

After you prepare the input as above, you can get output by running following commands.

```
python mhc_pseudo_matching.py I ./example/inputs/pmhc_im.csv ./example/inputs/pmhc_im_modified.csv \
python inference_csv.py --csv_path example/inputs/pmhc_im_modified.csv --inf_type pmhc_im_neo --output example/outputs/pmhc_im_output.csv
```

The output score represents the immunogenicity probability of the input peptide-MHC pair. This score can be used to rank the immunogenicity of peptide-MHC pairs or to determine if a pair is immunogenic by checking if the score exceeds 0.5. Both applications show excellent performance according to our manuscript benchmarks.

### p-mhc immunogenicity prediction for infectious disease vaccine development

For this category, the CSV file must include Allele and peptide columns.

Allele represents allele type of each MHC molecule. The allele types should be selected from the allele column of MHC_classI_pseudo.csv. While it is possible to use other allele names, the accuracy of the results cannot be guaranteed.

peptide represents peptide you want to check immunogenicity. Each character in the peptide column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports peptides up to a maximum length of 20 amino acids, with optimal performance observed with 9-mer peptides.

After you prepare the input as above, you can get output by running following commands.

```
python mhc_pseudo_matching.py I ./example/inputs/pmhc_im.csv ./example/inputs/pmhc_im_modified.csv \
python inference_csv.py --csv_path example/inputs/pmhc_im_modified.csv --inf_type pmhc_im_inf --output example/outputs/pmhc_im_output.csv
```

The output score represents the immunogenicity probability of the input peptide-MHC pair. This score can be used to rank the immunogenicity of peptide-MHC pairs or to determine if a pair is immunogenic by checking if the score exceeds 0.5. Both applications show excellent performance according to our manuscript benchmarks.

### ADA level prediction of protein drug sequences

For this category, CSV file must include peptide columns.

peptide represents peptide you want to check immunogenicity. Each character in the peptide column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports peptides up to a maximum length of 20 amino acids, with optimal performance observed with 9-mer peptides. If you want to input protein sequence longer than 9, we recommend you to cut off sequence into 9-mer peptides while sliding the window.

After you prepare the input as above, you can get output by running following commands.

```
python inference_csv.py --csv_path example/inputs/p_im.csv --inf_type p_im --output example/outputs/p_im_output.csv
```

The output score indicates the immunogenicity probability of the input peptide sequence. We recommend using this score to identify immunogenic parts of the sequence. If the score exceeds 0.5, the peptide is considered immunogenic. If over 20% of the peptides are immunogenic, the entire sequence can be classified as immunogenic. These thresholds are recommended based on our benchmarks.

### p-MHC binding classification (class I)

For this category, the CSV file must include Allele and peptide columns.

Allele represents allele type of each MHC molecule. The allele types should be selected from the allele column of MHC_classI_pseudo.csv. While it is possible to use other allele names, the accuracy of the results cannot be guaranteed.

peptide represents peptide you want to check. Each character in the peptide column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports peptides up to a maximum length of 20 amino acids, with optimal performance observed with 9-mer peptides.

After you prepare the input as above, you can get output by running following commands.

```
python mhc_pseudo_matching.py I ./example/inputs/pmhc_ba_I.csv ./example/inputs/pmhc_ba_I_modified.csv \
python inference_csv.py --csv_path example/inputs/pmhc_ba_I_modified.csv --inf_type pmhc_ba_I --output example/outputs/pmhc_ba_I_output.csv
```


The output score represents the binding probability of the input peptide-MHC class I pair. This score can be used to rank binding probabilities of peptide-MHC pairs or to determine if a pair is binding by checking if the score exceeds 0.5. Both applications show excellent performance according to our manuscript benchmarks.


### p-MHC binding classification (class II)

For this category, the CSV file must include Allele and peptide columns.

Allele represents allele type of each MHC molecule. The allele types should be selected from the allele column of MHC_classII_pseudo.csv. While it is possible to use other allele names, the accuracy of the results cannot be guaranteed.

peptide represents peptide you want to check. Each character in the peptide column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports peptides up to a maximum length of 20 amino acids, with optimal performance observed with 9-mer peptides.

After you prepare the input as above, you can get output by running following commands.

```
python mhc_pseudo_matching.py II ./example/inputs/pmhc_ba_II.csv ./example/inputs/pmhc_ba_II_modified.csv
python inference_csv.py --csv_path example/inputs/pmhc_ba_II_modified.csv --inf_type pmhc_ba_II --output example/outputs/pmhc_ba_II_output.csv
```

The output score represents the binding probability of the input peptide-MHC class II pair. This score can be used to rank binding probabilities of peptide-MHC pairs or to determine if a pair is binding by checking if the score exceeds 0.5. Both applications show excellent performance according to our manuscript benchmarks.

### TCR-p-MHC binding classification

For this category, the CSV file must include CDR3b and peptide columns.

CDR3b represents TCR's CDR 3β sequence. Each character in the CDR3b column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports sequence up to maximum length of 25 amino acids.

peptide represents peptide you want to check. Each character in the peptide column must be a single-letter representation of the 20 standard amino acids. Any deviation will result in an error. The model supports peptides up to a maximum length of 20 amino acids, with optimal performance observed with 9-mer peptides.

After you prepare the input as above, you can get output by running following commands.

```
python inference_csv.py --csv_path example/inputs/ptcr_ba.csv --inf_type ptcr_ba --output example/outputs/pmhc_ba_output.csv
```

The output score represents the binding probability of the input TCR CDR3 β-peptide pair. This score can be used to rank binding probabilities of TCR CDR3 β-peptide pairs or to determine if a pair is binding by checking if the score exceeds 0.5.

## License
All code including weight of neural network is licensed under the CC BY-NC-ND 4.0 license. 

