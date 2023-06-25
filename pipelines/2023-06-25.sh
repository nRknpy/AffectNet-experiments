python scripts/main_contrastive.py exp=CL_valence_z64_bs512
python scripts/main_finetuning.py exp=FT_valence_z64_bs512
python scripts/main_finetuning.py exp=FT_vanilla
python scripts/main_contrastive.py exp=CL_catval_z64_bs512
python scripts/main_finetuning.py exp=FT_catval_z64_bs512
python scripts/main_contrastive.py exp=CL_expression_z64_bs512
python scripts/main_finetuning.py exp=FT_expression_z64_bs512
