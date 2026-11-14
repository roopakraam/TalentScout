# Plan review with the CSE578 instructor (T0.5, T1.13)

This meeting has to happen in person, so the tracker keeps T0.5 and T1.13 open until
the approval is recorded at the bottom of this file.

## Email to send

> **Subject:** CSE578 project plan: TalentScout v2, 15-minute review before 22 Nov?
>
> Dear Sir/Madam,
>
> I have attached the plan for my CSE578 project, "Nature-Inspired Feature Selection for
> Football Talent Identification". It compares a Genetic Algorithm, Binary PSO and
> Binary Grey Wolf Optimizer for choosing the smallest set of player attributes that
> still predicts which young players reach an 80+ rating, using FIFA / EA FC data.
>
> Could I have 15 minutes before 22 November to get the plan approved? I have three
> questions:
>
> 1. Is an overlap of about 30% with the course content acceptable for the project?
> 2. Would you like the hybrid GA-PSO stretch goal (Unit VI, CO6) included?
> 3. What report format and page limit do you expect?
>
> Thank you,
> Roopak Raam

Attach `TalentScout_Research_Plan_v2_Optimization.md` (or a PDF of it).

## Five-minute summary for the meeting

1. **Problem.** EA rates every footballer on about 45 attributes. Which players aged 21 or
   under and rated below 80 reach 80+ within the next three game versions? A scout cannot
   measure 45 attributes for every youth player, so the optimization question is: what is
   the smallest attribute set that predicts this as well as all 45?
2. **Data and label.** Kaggle "EA Sports FC 24 Complete Player Dataset", 180,021 player-versions
   (FIFA 15 to FC 24). After filtering: 19,410 outfield rows with known outcome, 2.7% positive.
3. **Split.** Train FIFA 15–18, validation FIFA 19–20, test FIFA 21; every test player is
   removed from train and validation. The test set is used once at the end.
4. **Optimization.** Binary vector x in {0,1}^45, minimise 0.99·(1 − PR-AUC) + 0.01·|x|/45
   with a LightGBM wrapper. GA (Unit II), Binary PSO (Unit III), Binary GWO (Unit IV) and
   random search, each with 600 evaluations, the same starting population and 10 seeds;
   Wilcoxon tests with Holm correction; Bayesian optimization (Optuna TPE) against random
   search for hyperparameters (Unit V); an α sweep for the accuracy / size trade-off.
5. **Timeline.** Data frozen in week 1, optimizer runs in weeks 3–5, test evaluation and
   freeze on 27 December, report on 10 January.

## Record of the meeting

- Date:
- Changes requested:
- Answers to the three questions:
- Approval (paste the email or write "approved in person on ..."):
