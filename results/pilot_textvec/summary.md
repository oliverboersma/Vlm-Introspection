# Sweep summary: `results/pilot_textvec`

12896 probe records.

## Uninjected reference points (raw log-odds yes − no)

| condition | probe | on_target | mean | se | n |
|---|---|---|---|---|---|
| none | see | False | -2.49 | 0.07 | 72 |
| none | see | True | -2.49 | 0.13 | 24 |
| none | think | False | -0.81 | 0.03 | 72 |
| none | think | True | -0.81 | 0.06 | 24 |
| real | see | False | -3.28 | 0.18 | 72 |
| real | see | True | 3.50 | 0.27 | 24 |
| real | think | False | -1.17 | 0.11 | 72 |
| real | think | True | 0.85 | 0.07 | 24 |
| text_only_none | think | False | -0.03 | 0.15 | 24 |
| text_only_none | think | True | -0.03 | 0.26 | 8 |

## Real images of the concept (Δ vs neutral image)

| probe | on | off | spec | spec_se | n |
|---|---|---|---|---|---|
| see | 5.99 | -0.79 | 6.78 | 0.48 | 24 |
| think | 1.66 | -0.36 | 2.02 | 0.21 | 24 |

## Specificity (on − off-target Δ log-odds), α = 0.5

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 4.33 | 1.08 | -0.15 | -0.02 | 1.09 | 0.27 | -0.08 | -0.04 | 0.27 | 0.03 |
| 8 | 2.93 | 0.67 | -0.14 | -0.03 | 0.76 | 0.16 | 0.03 | 0.01 | 0.21 | -0.07 |
| 12 | 1.12 | 0.24 | 0.05 | 0.01 | 0.27 | 0.12 | -0.19 | -0.03 | 0.17 | 0.01 |
| 16 | 3.13 | 0.62 | -0.13 | -0.02 | 0.36 | -0.09 | -0.13 | 0.05 | -0.43 | 0.04 |
| 20 | 0.04 | 0.01 | 0.01 | 0.01 | 0.62 | 0.20 | 0.11 | 0.10 | 0.33 | 0.19 |

## Specificity (on − off-target Δ log-odds), α = 1

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 4.29 | 1.24 | -0.02 | 0.03 | 0.32 | -0.02 | 0.05 | -0.01 | -0.09 | 0.02 |
| 8 | 3.30 | 0.82 | -0.18 | -0.03 | 0.32 | 0.00 | -0.03 | 0.13 | -0.01 | 0.20 |
| 12 | 1.65 | 0.41 | 0.06 | 0.03 | 0.35 | 0.07 | -0.19 | -0.07 | 0.08 | -0.06 |
| 16 | 3.96 | 0.92 | -0.24 | -0.02 | 0.03 | -0.09 | -0.07 | 0.04 | -0.29 | -0.16 |
| 20 | 0.10 | 0.02 | 0.00 | 0.03 | 0.35 | 0.13 | 0.08 | 0.09 | 0.18 | 0.18 |

## Specificity (on − off-target Δ log-odds), α = 2

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.94 | 0.48 | 0.09 | -0.04 | -0.04 | -0.01 | -0.04 | 0.01 | -0.03 | 0.07 |
| 8 | 1.56 | 0.52 | -0.14 | -0.00 | 0.04 | -0.02 | -0.02 | -0.00 | -0.01 | -0.04 |
| 12 | 1.17 | 0.35 | -0.02 | 0.01 | 0.03 | -0.01 | 0.04 | 0.02 | -0.00 | -0.04 |
| 16 | 3.08 | 0.89 | -0.39 | -0.05 | 0.21 | 0.17 | 0.00 | 0.02 | 0.06 | 0.04 |
| 20 | 0.22 | 0.06 | -0.02 | 0.02 | 0.08 | 0.03 | 0.04 | -0.00 | 0.02 | 0.01 |

## Per concept at the best image-site setting (layer 4, α = 0.5)

| concept | see_on | see_off | think_on | think_off |
|---|---|---|---|---|
| red | 5.52 | -0.13 | 1.16 | -0.30 |
| blue | 2.53 | -0.05 | 0.80 | 0.13 |
| green | 4.19 | -0.43 | 1.22 | -0.00 |
| yellow | 1.47 | -0.97 | 0.25 | -0.49 |
| circle | 6.17 | -0.92 | 0.96 | -0.08 |
| square | 3.36 | -0.69 | 1.25 | 0.05 |
| triangle | 2.79 | -0.73 | 0.65 | -0.50 |
| star | 4.15 | -0.55 | 1.06 | -0.07 |
