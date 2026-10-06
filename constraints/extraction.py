import pandas as pd
import numpy as np
from itertools import permutations


def evaluate_response(trace, activation, target):
    """
    Response[A,B]

    Se A avviene, ogni occorrenza di A deve avere
    almeno un B successivamente.

    Se A non compare -> vacuous.
    """

    positions_a = [
        i for i, activity in enumerate(trace)
        if activity == activation
    ]

    if not positions_a:
        return "vacuous"

    for pos_a in positions_a:

        target_after = any(
            trace[j] == target
            for j in range(pos_a + 1, len(trace))
        )

        if not target_after:
            return "violated"

    return "fulfilled"


def evaluate_precedence(trace, activation, target):
    """
    Precedence[A,B]

    Ogni occorrenza di B deve avere almeno
    un A precedente.

    L'attivazione logica del constraint è B.

    Se B non compare -> vacuous.
    """

    positions_b = [
        i for i, activity in enumerate(trace)
        if activity == target
    ]

    if not positions_b:
        return "vacuous"

    for pos_b in positions_b:

        activation_before = any(
            trace[j] == activation
            for j in range(pos_b)
        )

        if not activation_before:
            return "violated"

    return "fulfilled"


def evaluate_chain_response(trace, activation, target):
    """
    Chain Response[A,B]

    Ogni volta che compare A, l'evento
    immediatamente successivo deve essere B.

    Se A non compare -> vacuous.
    """

    positions_a = [
        i for i, activity in enumerate(trace)
        if activity == activation
    ]

    if not positions_a:
        return "vacuous"

    for pos_a in positions_a:

        # A è l'ultimo evento della trace
        if pos_a + 1 >= len(trace):
            return "violated"

        if trace[pos_a + 1] != target:
            return "violated"

    return "fulfilled"


EVALUATORS = {
    "Response": evaluate_response,
    "Precedence": evaluate_precedence,
    "Chain Response": evaluate_chain_response,
}




def evaluate_constraint(
    traces,
    template,
    activation,
    target
):

    evaluator = EVALUATORS[template]

    n_fulfilled = 0
    n_violated = 0
    n_vacuous = 0

    for trace in traces.values():

        state = evaluator(
            trace,
            activation,
            target
        )

        if state == "fulfilled":
            n_fulfilled += 1

        elif state == "violated":
            n_violated += 1

        elif state == "vacuous":
            n_vacuous += 1

    n_total = len(traces)

    n_activated = (
        n_fulfilled +
        n_violated
    )

    # --------------------------------------------------------
    # Activation rate
    #
    # Percentuale di case in cui il constraint
    # viene effettivamente attivato.
    # --------------------------------------------------------

    activation_rate = (
        n_activated / n_total
        if n_total > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Violation rate sulle trace attivate
    # --------------------------------------------------------

    violation_rate = (
        n_violated / n_activated
        if n_activated > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Non-vacuous compliance
    #
    # Consideriamo SOLO le trace in cui il constraint
    # è stato attivato.
    # --------------------------------------------------------

    non_vacuous_compliance = (
        n_fulfilled / n_activated
        if n_activated > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Compliance
    #
    # Consideriamo le trace in cui il constraint è soddisfatto (vacue + fulfilled) 
    # sulle totali
    # 
    # --------------------------------------------------------

    compliance = (
        (n_fulfilled  + n_vacuous)/ n_total
        if n_activated > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Vacuity rate
    # --------------------------------------------------------

    vacuity_rate = (
        n_vacuous / n_total
        if n_total > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Violation coverage
    #
    # Percentuale dell'intero dataset che viola
    # effettivamente il constraint.
    # --------------------------------------------------------

    violation_coverage = (
        n_violated / n_total
        if n_total > 0
        else np.nan
    )




    return {
        "template": template,
        "activation": activation,
        "target": target,
        "constraint": ( f"{template}[{activation},{target}]"),
        "n_cases": n_total,
        "n_activated": n_activated,
        "n_fulfilled": n_fulfilled,
        "n_violated": n_violated,
        "n_vacuous": n_vacuous,
        "activation_rate": activation_rate,
        "non_vacuous_compliance":  non_vacuous_compliance,
        "compliance": compliance,
        "violation_rate":violation_rate,
        "violation_coverage": violation_coverage,
        "vacuity_rate": vacuity_rate
    } #evaluation di un singolo vincolo rispetto a tutto il log



def main():
    log_name= "bpic2017_o"
    df = pd.read_csv("event_log/"+ log_name + ".csv")


    # Template DECLARE da analizzare
    TEMPLATES = [
        "Response", #se avviene A, B deve avvenire successivamente 
        "Precedence", #B può avvenire solo se A è avvenuto prima
        "Chain Response", #se avviene A l'elemento immediatamente successivo è B
    ]

    # Filtri per selezionare vincoli interessanti
    MIN_NON_VACUOUS_COMPLIANCE = 0.70
    MAX_NON_VACUOUS_COMPLIANCE = 0.95

    MIN_ACTIVATION_RATE = 0.20
    MIN_VIOLATION_COVERAGE = 0.01
    MIN_ACTIVATED_CASES = 50

    # Bisogna calcolare:
    # n_cases -> numero di tracce nel dataset
    # n_activated -> numero di tracce in cui vincolo è attivato
    # n_fulfilled -> numero di tracce in cui il vincolo è attivato e rispettato
    # n_violated -> numero di tracce in cui il vincolo è attivato e violato
    # n_vacuous -> numero di tracce in cui il vincolo non viene attivato
    # n_activated= n_fulfilled + n_violated

    # activation_rate= n_activated/n_cases
    # non_vacuous_compliance -> n_fulfilled/n_activated
    # violation_coverage -> n_violated/n_cases

    # File di output
    OUTPUT_ALL = "constraints/outputs/"+log_name+"_all.csv"
    OUTPUT_SELECTED = "constraints/outputs/"+log_name+"_selected.csv"



    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["case", "timestamp"]
    ).reset_index(drop=True)

    print(f"Events: {len(df)}")
    print(f"Cases: {df['case'].nunique()}")
    print(f"Activities: {df['activity'].nunique()}")


    traces = (
        df
        .groupby("case", sort=False)["activity"]
        .apply(list)
        .to_dict()
    )#dizionario case -> list di activity ordinate


    activities = sorted(
        df["activity"]
        .dropna()
        .unique()
    )



    #GENERAZIONE DEI VINCOLI
    # Generiamo tutte le coppie ordinate A,B.
    activity_pairs = list(
        permutations(activities, 2)
    )

    print(f"\nCoppie di activity da analizzare:" f"{len(activity_pairs)}")
    print(f"Constraint totali da valutare: "f"{len(activity_pairs) * len(TEMPLATES)}")

    results = []

    for template in TEMPLATES:
        print(
            f"\nAnalisi template: {template}"
        )

        for i, (activation, target) in enumerate(activity_pairs, start=1):

            metrics = evaluate_constraint(
                traces=traces,
                template=template,
                activation=activation,
                target=target
            )
            results.append(metrics)


    constraints_df = pd.DataFrame(results)
    constraints_df = constraints_df[ constraints_df["n_activated"] > 0].copy()


    selected_df = constraints_df[
        (
            constraints_df["non_vacuous_compliance"]
            >= MIN_NON_VACUOUS_COMPLIANCE
        )
        &
        (
            constraints_df["non_vacuous_compliance"]
            <= MAX_NON_VACUOUS_COMPLIANCE
        )
        &
        (
            constraints_df["activation_rate"]
            >= MIN_ACTIVATION_RATE
        )
        &
        (
            constraints_df["violation_coverage"]
            >= MIN_VIOLATION_COVERAGE
        )
        &
        (
            constraints_df["n_activated"]
            >= MIN_ACTIVATED_CASES
        )
    ].copy()


    # ORDINAMENTO
    # 1. activation rate elevato
    # 2. compliance elevata
    # 3. molte violazioni osservabili

    selected_df = selected_df.sort_values(
        [   "activation_rate", "non_vacuous_compliance", "violation_coverage", ],
        ascending=[ False, False, False]
    )

    constraints_df.to_csv( OUTPUT_ALL, index=False)

    selected_df.to_csv(OUTPUT_SELECTED, index=False)



    print("\n" + "=" * 80)

    print( f"Constraint valutati: "f"{len(constraints_df)}")

    print(f"Constraint selezionati: " f"{len(selected_df)}")

    print("=" * 80)


    columns_to_show = [ "constraint","compliance","non_vacuous_compliance", "violation_rate", "activation_rate", "vacuity_rate",
                     # "n_activated", "n_fulfilled", "n_violated", "n_vacuous"
                     ]


    if len(selected_df) > 0:

        print("\nTOP CONSTRAINT:\n")

        print(
            selected_df[
                columns_to_show
            ]
            .head(30)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    else:

        print(
            "\nNessun constraint soddisfa "
            "i filtri impostati."
        )


    print(
        f"\nTutti i constraint salvati in: "
        f"{OUTPUT_ALL}"
    )

    print(
        f"Constraint selezionati salvati in: "
        f"{OUTPUT_SELECTED}"
    )


if __name__=="__main__":
    main()