import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

from src.io_utils_fused import CSVDataset_test,Collater_test
from src.model_fused import Finaltask1_perf, task3, task4, task6, task7, task9

from src.constants import PAD,PROTEIN_ALPHABET

from src.utils_fused import calculate_auroc, calculate_accuracy, calculate_f1_score, calculate_precision_recall
import glob

parent = Path(__file__).resolve(True).parent


def main(args):
    if args.inf_type == "pmhc_im_neo":
        name_ = "best_param/pmhc_im_neo/BigMHC_finalMedium_OAS_el-mlm_ADV1.0_bestvalloss.pt"
    elif args.inf_type == "pmhc_im_inf":
        name_ = "best_param/pmhc_im_inf/BigMHC_finalfinetune-Small_OAS_el-fused_ADV1.0_bestvalauprc.pt"
    elif args.inf_type == "p_im":
        name_ = "best_param/p_im_ada/finalSmall_OAS_el-neg_ADV1.0_bestvalloss.pt"
    elif args.inf_type == "pmhc_ba_I":
        name_ = "best_param/pmhc_ba_I/Small_OAS_el-fused_ADV1.0_60.pt"
    elif args.inf_type == "pmhc_ba_II":
        name_ = "best_param/pmhc_ba_II/Uni_el-mlm_ADV1.0_95.pt"
    elif args.inf_type == "ptcr_ba":
        name_ = "best_param/ptcr_ba_neo/Uni_el-fused_ADV1.0_0.pt"
    else:
        raise ValueError(f"Unknown inf_type: {args.inf_type!r}")

    args.name = name_
    save_path = parent / name_
    if ("rnd" in name_) | ("fused" in name_):
        args.d_model = 280
        args.embedding_dim = 280
    else:
        args.d_model = 300
        args.embedding_dim = 300
    model_final = Finaltask1_perf(
        d_model=args.d_model,
        n_tokens=29,
        kernel_size=1,
        n_layers=6,
        d_embedding=args.embedding_dim,
        r=1,
        mask_condition = False
    )
    device = torch.device("cpu")

    ckpt = torch.load(save_path, map_location=device)
    
    if (args.inf_type == "pmhc_im") | (args.inf_type == "p_im"):
        model_final = Finaltask1_perf(
            d_model=args.d_model,
            n_tokens=29,
            kernel_size=1,
            n_layers=6,
            d_embedding=args.embedding_dim,
            r=1,
            mask_condition = False
        )
        model_state_dict = ckpt["model_state_dict"]
        model_final.load_state_dict(model_state_dict)
    if args.inf_type == "pmhc_ba_I":
        if "fused" in args.name:
            from src.model_fused import task3
            model_final = task3(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1,
                mask_condition = False
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_encoder.', ''): v for k, v in model_state_dict.items() if 'task3_encoder.' in k}
            model_final.task3_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_decoder.', ''): v for k, v in model_state_dict.items() if 'task3_decoder.' in k}
            model_final.task3_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "neg" in args.name:
            from src.model_el import task3
            model_final = task3(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1,
                mask_condition = False
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_encoder.', ''): v for k, v in model_state_dict.items() if 'task3_encoder.' in k}
            model_final.task3_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_decoder.', ''): v for k, v in model_state_dict.items() if 'task3_decoder.' in k}
            model_final.task3_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "mlm" in args.name:
            from src.model_mlm import task3
            model_final = task3(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1,
                mask_condition = False
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_encoder.', ''): v for k, v in model_state_dict.items() if 'task3_encoder.' in k}
            model_final.task3_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_decoder.', ''): v for k, v in model_state_dict.items() if 'task3_decoder.' in k}
            model_final.task3_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
    if args.inf_type == "pmhc_el_I":
        if "fused" in args.name:
            from src.model_fused import task4
            model_final = task4(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task4_encoder.', ''): v for k, v in model_state_dict.items() if 'task4_encoder.' in k}
            model_final.task4_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task4_decoder.', ''): v for k, v in model_state_dict.items() if 'task4_decoder.' in k}
            model_final.task4_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "neg" in args.name:
            from src.model_el import task4
            model_final = task4(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task3_encoder.', ''): v for k, v in model_state_dict.items() if 'task3_encoder.' in k}
            model_final.task4_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task4_decoder.', ''): v for k, v in model_state_dict.items() if 'task4_decoder.' in k}
            model_final.task4_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "mlm" in args.name:
            print("NO MLM MODEL FOR EL")
            return
    if args.inf_type == "pmhc_ba_II":
        if "fused" in args.name:
            from src.model_fused import task6
            model_final = task6(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1,
                mask_condition = False
            )
            model_state_dict = ckpt
            
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)    
            adjusted_state_dict = {k.replace('task6_encoder.', ''): v for k, v in model_state_dict.items() if 'task6_encoder.' in k}
            model_final.task6_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task6_decoder.', ''): v for k, v in model_state_dict.items() if 'task6_decoder.' in k}
            model_final.task6_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "neg" in args.name:
            from src.model_el import task5
            model_final = task5(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task5_encoder.', ''): v for k, v in model_state_dict.items() if 'task5_encoder.' in k}
            model_final.task5_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task5_decoder.', ''): v for k, v in model_state_dict.items() if 'task5_decoder.' in k}
            model_final.task5_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "mlm" in args.name:
            from src.model_mlm import task5
            model_final = task5(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task5_encoder.', ''): v for k, v in model_state_dict.items() if 'task5_encoder.' in k}
            model_final.task5_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task5_decoder.', ''): v for k, v in model_state_dict.items() if 'task5_decoder.' in k}
            model_final.task5_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
    if args.inf_type == "pmhc_el_II":
        if "fused" in args.name:
            from src.model_fused import task7
            model_final = task7(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1,
                mask_condition = False
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task7_encoder.', ''): v for k, v in model_state_dict.items() if 'task7_encoder.' in k}
            model_final.task7_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task7_decoder.', ''): v for k, v in model_state_dict.items() if 'task7_decoder.' in k}
            model_final.task7_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "neg" in args.name:
            from src.model_el import task6
            model_final = task6(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task5_encoder.', ''): v for k, v in model_state_dict.items() if 'task5_encoder.' in k}
            model_final.task6_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task6_decoder.', ''): v for k, v in model_state_dict.items() if 'task6_decoder.' in k}
            model_final.task6_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "mlm " in args.name:
            print("NO MLM MODEL FOR EL")
            return
    if args.inf_type == "ptcr_ba":
        if "fused" in args.name:
            from src.model_fused import task9
            model_final = task9(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1,
                mask_condition = False
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task9_encoder.', ''): v for k, v in model_state_dict.items() if 'task9_encoder.' in k}
            model_final.task9_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task9_decoder.', ''): v for k, v in model_state_dict.items() if 'task9_decoder.' in k}
            model_final.task9_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "neg" in args.name:
            from src.model_el import task7
            model_final = task7(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task7_encoder.', ''): v for k, v in model_state_dict.items() if 'task7_encoder.' in k}
            model_final.task7_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task7_decoder.', ''): v for k, v in model_state_dict.items() if 'task7_decoder.' in k}
            model_final.task7_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
        elif "mlm" in args.name:
            from src.model_mlm import task7
            model_final = task7(
                d_model=args.d_model,
                n_tokens=29,
                kernel_size=1,
                n_layers=6,
                d_embedding=args.embedding_dim,
                r=1
            )
            model_state_dict = ckpt
            adjusted_state_dict = {k.replace('shared_encoder.', ''): v for k, v in model_state_dict.items() if 'shared_encoder.' in k}
            model_final.shared_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task7_encoder.', ''): v for k, v in model_state_dict.items() if 'task7_encoder.' in k}
            model_final.task7_encoder.load_state_dict(adjusted_state_dict)
            adjusted_state_dict = {k.replace('task7_decoder.', ''): v for k, v in model_state_dict.items() if 'task7_decoder.' in k}
            model_final.task7_decoder.load_state_dict(adjusted_state_dict)
            model_final.to(device)
    model_final.eval()
    
    inf_collator = Collater_test(alphabet=PROTEIN_ALPHABET, pad=True, backwards=False, pad_token=PAD)
    df = pd.read_csv(args.csv_path)
    # preprocess df 
    print("Preprocessing the data")
    # add CDR3b column and empty data if there are no CDR3b
    df["CDR3b"] =""
    if "CDR3b" not in df.columns:
        df["CDR3b"] =""
    if "task" not in df.columns:
        df["task"] = [1]*len(df)
    if "pseudo" not in df.columns:
        df["pseudo"] = ""
    if "mhc" not in df.columns:
        df["mhc"] = ""
    if "label" not in df.columns:
        df["label"] = ""
    if "pep_seq" in df.columns:
        #change pep_seq to peptide
        df["peptide"] = df["pep_seq"]
    print("Data Preprocessing is done")
    test_dataset = CSVDataset_test(df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=inf_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_final.to(device)
    if "fused" in args.name:
        task_dict = {"pmhc_im":[3], "p_im":[1], "pmhc_el_I":[4], "pmhc_el_II":[7], "pmhc_ba_I":[3], "pmhc_ba_II":[6], "ptcr_ba":[9]}
    if "neg" in args.name:
        task_dict = {"pmhc_im":[3], "p_im":[1], "pmhc_el_I":[4], "pmhc_el_II":[6], "pmhc_ba_I":[3], "pmhc_ba_II":[5], "ptcr_ba":[7]}
    if "mlm" in args.name:
        task_dict = {"pmhc_im":[3], "p_im":[1], "pmhc_el_I":[4], "pmhc_el_II":[6], "pmhc_ba_I":[3], "pmhc_ba_II":[5], "ptcr_ba":[7]}
    with torch.no_grad():
        outputs = []
        fractions =[]
        #print(test_loader[0])
        for i,(src, m1, m2, tcr, frac, p_lens, mhcs) in enumerate(test_loader):
            src = src.to(device)
            m1, m2, tcr = m1.to(device), m2.to(device), tcr.to(device)
            frac = torch.FloatTensor(frac).unsqueeze(-1).to(device)
            
            output = model_final(src,m1,m2, tcr=tcr, task = task_dict[args.inf_type])
            output = output[-1]
            fractions.extend(frac.detach().cpu())
            outputs.extend(output.detach().cpu())
            print(f'Processed {i+1}/{len(test_loader)}')
        fractions = torch.stack(fractions, dim=0)
        outputs = torch.stack(outputs, dim=0)
        #df['pep_seq'] = df['peptide']
        #drop out label, cdr3b, task, pseudo, mhc
        if args.inf_type == "p_im":
            df = df.drop(columns = ['label', 'CDR3b', 'task', 'pseudo', 'mhc'])
        elif args.inf_type == "ptcr_ba":
            df = df.drop(columns = ['label', 'CDR3b', 'task', 'pseudo', 'mhc'])
        else:
            df = df.drop(columns = ['label', 'CDR3b', 'task', 'pseudo', 'mhc', 'allele'])
        
        df["score"] = outputs.numpy()
        df.to_csv(args.output, index = False)
        '''
        auroc = calculate_auroc(fractions, outputs)
        acc = calculate_accuracy(fractions, outputs)
        f1 = calculate_f1_score(fractions, outputs)
        precision, recall, auprc = calculate_precision_recall(fractions, outputs)
        paired = list(zip(outputs, fractions))
        sorted_by_prob = sorted(paired, key=lambda x: x[0], reverse=True)
        
        ppvn_values = []
        true_positive_count = 0
        # count number of 1.0 in true_labels
        true_count = sum(1 for x in fractions if x == 1)
        for n in range(1, true_count + 1):
            top_n = sorted_by_prob[:n]
            true_positive_count = sum(1 for _, actual in top_n if actual == 1)
            ppvn = true_positive_count / n
            ppvn_values.append(ppvn)
        
        mean_ppvn = np.mean(ppvn_values)
        print(f'AUROC: {auroc:.3f}')
        print(f'Accuracy: {acc:.3f}')
        print(f'F1: {f1:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'AUPRC: {auprc:.3f}')
        print(f'Mean PPVN: {mean_ppvn:.3f}')
        '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',type=str,required=True,
                        help = 'path to the csv file')
    parser.add_argument('--inf_type',type=str,required=True,
                        help = 'type of inference, pmhc_im, p_im, pmhc_ba_I, pmhc_ba_II, ptcr_ba ')
    parser.add_argument('--output',type=str,required=True,
                        help = 'path to the output csv file')
    args = parser.parse_args()

    print("Arguments:")
    for p in vars(args).items():
        print("  ",p[0]+": ",p[1])
    main(args)
