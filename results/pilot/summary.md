# Sweep summary: `results/pilot`

20384 probe records.

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

## Specificity (on − off-target Δ log-odds), α = 0.25

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.65 | 0.17 | -0.12 | -0.02 | 0.28 | 0.11 | 0.27 | 0.04 | 0.20 | -0.08 |
| 8 | 0.42 | 0.11 | -0.09 | -0.01 | 0.67 | 0.29 | 0.06 | 0.05 | 0.35 | -0.06 |
| 12 | 0.18 | 0.04 | 0.01 | 0.00 | 0.09 | 0.03 | -0.16 | -0.04 | 0.07 | -0.04 |
| 16 | 0.19 | 0.06 | -0.03 | -0.01 | 0.39 | 0.16 | -0.03 | -0.06 | 0.28 | -0.10 |
| 20 | 0.01 | 0.00 | 0.01 | 0.00 | 0.13 | 0.03 | 0.09 | 0.08 | 0.09 | 0.14 |
| 24 | -0.00 | -0.00 | 0.00 | 0.00 | 0.02 | 0.02 | -0.04 | 0.00 | 0.02 | 0.00 |

## Specificity (on − off-target Δ log-odds), α = 0.5

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 2.37 | 0.56 | -0.15 | -0.02 | 0.57 | 0.14 | -0.08 | -0.04 | 0.23 | 0.03 |
| 8 | 1.39 | 0.34 | -0.14 | -0.03 | 0.76 | 0.14 | 0.03 | 0.01 | 0.13 | -0.07 |
| 12 | 0.48 | 0.10 | 0.05 | 0.01 | 0.11 | 0.02 | -0.19 | -0.03 | 0.09 | 0.01 |
| 16 | 1.29 | 0.28 | -0.13 | -0.02 | 0.30 | 0.04 | -0.13 | 0.05 | 0.15 | 0.04 |
| 20 | 0.01 | 0.00 | 0.01 | 0.01 | 0.11 | 0.05 | 0.11 | 0.10 | 0.16 | 0.19 |
| 24 | -0.00 | -0.00 | 0.00 | -0.00 | 0.03 | 0.03 | -0.10 | -0.01 | 0.04 | -0.01 |

## Specificity (on − off-target Δ log-odds), α = 1

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 3.00 | 0.74 | -0.02 | 0.03 | -0.05 | -0.00 | 0.05 | -0.01 | 0.01 | 0.02 |
| 8 | 2.38 | 0.52 | -0.18 | -0.03 | 0.08 | -0.09 | -0.03 | 0.13 | -0.05 | 0.20 |
| 12 | 0.67 | 0.20 | 0.06 | 0.03 | 0.12 | 0.04 | -0.19 | -0.07 | 0.07 | -0.06 |
| 16 | 2.43 | 0.60 | -0.24 | -0.02 | -0.03 | -0.05 | -0.07 | 0.04 | -0.25 | -0.16 |
| 20 | 0.03 | 0.01 | 0.00 | 0.03 | 0.12 | -0.05 | 0.08 | 0.09 | 0.04 | 0.18 |
| 24 | 0.01 | -0.01 | 0.00 | -0.00 | 0.06 | 0.03 | -0.18 | -0.03 | 0.01 | -0.06 |

## Specificity (on − off-target Δ log-odds), α = 2

| layer | img_see | img_think | img_rand_see | img_rand_think | txt_see | txt_think | txt_rand_see | txt_rand_think | txtonly_think | txtonly_rand_think |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.06 | 0.17 | 0.09 | -0.04 | 0.01 | 0.01 | -0.04 | 0.01 | 0.04 | 0.07 |
| 8 | 1.05 | 0.33 | -0.14 | -0.00 | -0.02 | -0.00 | -0.02 | -0.00 | -0.00 | -0.04 |
| 12 | 0.35 | -0.02 | -0.02 | 0.01 | 0.02 | -0.04 | 0.04 | 0.02 | -0.05 | -0.04 |
| 16 | 2.98 | 1.10 | -0.39 | -0.05 | 0.24 | 0.01 | 0.00 | 0.02 | 0.00 | 0.04 |
| 20 | 0.08 | 0.04 | -0.02 | 0.02 | 0.09 | -0.04 | 0.04 | -0.00 | -0.04 | 0.01 |
| 24 | 0.01 | 0.01 | -0.00 | -0.01 | -0.01 | -0.02 | -0.06 | -0.00 | -0.05 | -0.02 |

## Per concept at the best image-site setting (layer 4, α = 1)

| concept | see_on | see_off | think_on | think_off |
|---|---|---|---|---|
| red | 5.59 | -0.40 | -0.53 | -0.88 |
| blue | 3.64 | -0.23 | 1.32 | 0.15 |
| green | 4.60 | 0.64 | 1.03 | -0.14 |
| yellow | 0.97 | -0.54 | 0.13 | -0.20 |
| circle | 2.08 | -0.33 | 0.37 | -0.29 |
| square | 0.48 | 1.17 | -0.10 | 0.05 |
| triangle | 1.20 | -0.58 | 0.33 | -0.44 |
| star | 3.34 | -1.86 | 0.64 | -1.01 |

## Free text: rate at which the injected concept is named

| condition | site | layer | alpha | question | rate | n |
|---|---|---|---|---|---|---|
| concept | image | 12 | 1.00 | describe | 0 | 4 |
| concept | image | 12 | 1.00 | intrusion | 0 | 4 |
| concept | image | 12 | 2.00 | describe | 0 | 4 |
| concept | image | 12 | 2.00 | intrusion | 0 | 4 |
| concept | image | 16 | 1.00 | describe | 0.25 | 4 |
| concept | image | 16 | 1.00 | intrusion | 0 | 4 |
| concept | image | 16 | 2.00 | describe | 0 | 4 |
| concept | image | 16 | 2.00 | intrusion | 0 | 4 |
| concept | image | 8 | 1.00 | describe | 0 | 4 |
| concept | image | 8 | 1.00 | intrusion | 0 | 4 |
| concept | image | 8 | 2.00 | describe | 0 | 4 |
| concept | image | 8 | 2.00 | intrusion | 0 | 4 |
| concept | text | 12 | 1.00 | describe | 0 | 4 |
| concept | text | 12 | 1.00 | intrusion | 0 | 4 |
| concept | text | 12 | 2.00 | describe | 0 | 4 |
| concept | text | 12 | 2.00 | intrusion | 0 | 4 |
| concept | text | 16 | 1.00 | describe | 0 | 4 |
| concept | text | 16 | 1.00 | intrusion | 0 | 4 |
| concept | text | 16 | 2.00 | describe | 0 | 4 |
| concept | text | 16 | 2.00 | intrusion | 0 | 4 |
| concept | text | 8 | 1.00 | describe | 0 | 4 |
| concept | text | 8 | 1.00 | intrusion | 0 | 4 |
| concept | text | 8 | 2.00 | describe | 0 | 4 |
| concept | text | 8 | 2.00 | intrusion | 0 | 4 |
| none | none | -1 | 0.00 | describe | 0 | 4 |
| none | none | -1 | 0.00 | intrusion | 0 | 4 |
| random | image | 12 | 1.00 | describe | 0 | 4 |
| random | image | 12 | 1.00 | intrusion | 0 | 4 |
| random | image | 12 | 2.00 | describe | 0 | 4 |
| random | image | 12 | 2.00 | intrusion | 0 | 4 |
| random | image | 16 | 1.00 | describe | 0 | 4 |
| random | image | 16 | 1.00 | intrusion | 0 | 4 |
| random | image | 16 | 2.00 | describe | 0 | 4 |
| random | image | 16 | 2.00 | intrusion | 0 | 4 |
| random | image | 8 | 1.00 | describe | 0 | 4 |
| random | image | 8 | 1.00 | intrusion | 0 | 4 |
| random | image | 8 | 2.00 | describe | 0 | 4 |
| random | image | 8 | 2.00 | intrusion | 0 | 4 |
| random | text | 12 | 1.00 | describe | 0 | 4 |
| random | text | 12 | 1.00 | intrusion | 0 | 4 |
| random | text | 12 | 2.00 | describe | 0 | 4 |
| random | text | 12 | 2.00 | intrusion | 0 | 4 |
| random | text | 16 | 1.00 | describe | 0 | 4 |
| random | text | 16 | 1.00 | intrusion | 0 | 4 |
| random | text | 16 | 2.00 | describe | 0 | 4 |
| random | text | 16 | 2.00 | intrusion | 0 | 4 |
| random | text | 8 | 1.00 | describe | 0 | 4 |
| random | text | 8 | 1.00 | intrusion | 0 | 4 |
| random | text | 8 | 2.00 | describe | 0 | 4 |
| random | text | 8 | 2.00 | intrusion | 0 | 4 |
