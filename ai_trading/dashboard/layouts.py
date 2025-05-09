"""
Layouts pour le dashboard interactif d'analyse post-trade.

Ce module définit les layouts et composants de l'interface utilisateur
du dashboard d'analyse post-trade.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_layout(app):
    """
    Crée le layout principal de l'application.

    Args:
        app: Instance de l'application Dash

    Returns:
        Layout principal
    """
    return html.Div(
        [
            # Store pour les données partagées entre les composants
            dcc.Store(id="shared-data", storage_type="memory"),
            # Header
            create_header(),
            # Navigation Tabs
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Vue d'ensemble",
                        tab_id="tab-overview",
                        children=[create_overview_layout()],
                    ),
                    dbc.Tab(
                        label="Performances",
                        tab_id="tab-performance",
                        children=[create_performance_layout()],
                    ),
                    dbc.Tab(
                        label="Transactions",
                        tab_id="tab-transactions",
                        children=[create_transactions_layout()],
                    ),
                    dbc.Tab(
                        label="Allocation",
                        tab_id="tab-allocation",
                        children=[create_allocation_layout()],
                    ),
                    dbc.Tab(
                        label="Risque",
                        tab_id="tab-risk",
                        children=[create_risk_layout()],
                    ),
                    dbc.Tab(
                        label="Facteurs",
                        tab_id="tab-factors",
                        children=[create_factors_layout()],
                    ),
                    dbc.Tab(
                        label="Visualisations 3D",
                        tab_id="tab-viz3d",
                        children=[create_viz3d_layout()],
                    ),
                    dbc.Tab(
                        label="Analyse Post-Mortem",
                        tab_id="tab-postmortem",
                        children=[create_postmortem_layout()],
                    ),
                ],
                id="tabs",
                active_tab="tab-overview",
            ),
            # Footer
            create_footer(),
        ]
    )


def create_header():
    """
    Crée le header du dashboard.

    Returns:
        Component Header
    """
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(src="/assets/logo.png", height="50px"),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "Crypto Trading - Analyse Post-Trade",
                                    className="ms-2",
                                )
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Documentation", href="#")),
                            dbc.NavItem(dbc.NavLink("API", href="#")),
                            dbc.DropdownMenu(
                                children=[
                                    dbc.DropdownMenuItem(
                                        "Portefeuille Principal", id="portfolio-main"
                                    ),
                                    dbc.DropdownMenuItem(
                                        "Portefeuille Test", id="portfolio-test"
                                    ),
                                    dbc.DropdownMenuItem(
                                        "Portefeuille Backtest", id="portfolio-backtest"
                                    ),
                                ],
                                nav=True,
                                in_navbar=True,
                                label="Portefeuille",
                            ),
                        ]
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]
        ),
        color="primary",
        dark=True,
        className="mb-4",
    )


def create_footer():
    """
    Crée le footer du dashboard.

    Returns:
        Component Footer
    """
    return html.Footer(
        dbc.Container(
            [
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.P(
                                "© 2023 Crypto Trading AI System",
                                className="text-center",
                            )
                        ),
                    ]
                ),
            ]
        ),
        className="mt-4",
    )


def create_overview_layout():
    """
    Crée le layout pour la vue d'ensemble.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Vue d'ensemble du Portefeuille"),
                            html.P(id="portfolio-date-range", className="lead"),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Cartes des métriques clés
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Valeur du Portefeuille",
                                            className="card-title",
                                        ),
                                        html.H2(
                                            id="overview-value",
                                            className="card-value text-primary",
                                        ),
                                        html.P(
                                            id="overview-value-change",
                                            className="card-change",
                                        ),
                                    ]
                                )
                            ]
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Performance Totale", className="card-title"
                                        ),
                                        html.H2(
                                            id="overview-performance",
                                            className="card-value text-success",
                                        ),
                                        html.P(
                                            id="overview-performance-period",
                                            className="card-change",
                                        ),
                                    ]
                                )
                            ]
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Ratio de Sharpe", className="card-title"
                                        ),
                                        html.H2(
                                            id="overview-sharpe",
                                            className="card-value text-info",
                                        ),
                                        html.P(
                                            "Rendement ajusté au risque",
                                            className="card-description",
                                        ),
                                    ]
                                )
                            ]
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Drawdown Max", className="card-title"),
                                        html.H2(
                                            id="overview-drawdown",
                                            className="card-value text-danger",
                                        ),
                                        html.P(
                                            id="overview-drawdown-date",
                                            className="card-description",
                                        ),
                                    ]
                                )
                            ]
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Graphique principal
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Évolution de la Valeur du Portefeuille"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="overview-portfolio-chart"),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "1M",
                                                        id="btn-timeframe-1m",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                    dbc.Button(
                                                        "3M",
                                                        id="btn-timeframe-3m",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                    dbc.Button(
                                                        "6M",
                                                        id="btn-timeframe-6m",
                                                        color="secondary",
                                                        outline=True,
                                                        active=True,
                                                    ),
                                                    dbc.Button(
                                                        "1A",
                                                        id="btn-timeframe-1y",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                    dbc.Button(
                                                        "Max",
                                                        id="btn-timeframe-max",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                ],
                                                className="mt-2",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Allocation Actuelle")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="overview-allocation-chart"),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Activité Récente")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="overview-recent-activity"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Résumé des transactions et de la performance
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Transactions Récentes")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="overview-recent-transactions"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Performances par Actif")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="overview-asset-performance"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


def create_performance_layout():
    """
    Crée le layout pour l'analyse de performance.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Analyse de Performance"),
                            html.P(
                                "Visualisation détaillée des performances du portefeuille",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Contrôles et Filtres
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Période d'analyse"
                                                            ),
                                                            dcc.DatePickerRange(
                                                                id="performance-date-range",
                                                                start_date_placeholder_text="Date de début",
                                                                end_date_placeholder_text="Date de fin",
                                                                calendar_orientation="horizontal",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Benchmark"),
                                                            dcc.Dropdown(
                                                                id="performance-benchmark",
                                                                options=[
                                                                    {
                                                                        "label": "Bitcoin (BTC)",
                                                                        "value": "BTC",
                                                                    },
                                                                    {
                                                                        "label": "Ethereum (ETH)",
                                                                        "value": "ETH",
                                                                    },
                                                                    {
                                                                        "label": "Crypto Total Market Cap",
                                                                        "value": "TOTAL",
                                                                    },
                                                                    {
                                                                        "label": "S&P 500",
                                                                        "value": "SPX",
                                                                    },
                                                                    {
                                                                        "label": "Aucun",
                                                                        "value": "NONE",
                                                                    },
                                                                ],
                                                                value="BTC",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Fréquence"),
                                                            dcc.Dropdown(
                                                                id="performance-frequency",
                                                                options=[
                                                                    {
                                                                        "label": "Journalière",
                                                                        "value": "D",
                                                                    },
                                                                    {
                                                                        "label": "Hebdomadaire",
                                                                        "value": "W",
                                                                    },
                                                                    {
                                                                        "label": "Mensuelle",
                                                                        "value": "M",
                                                                    },
                                                                ],
                                                                value="D",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Graphiques de performance
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Performance Cumulée")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="performance-cumulative-chart"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Rendements Périodiques")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="performance-periodic-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Comparaison au Benchmark")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="performance-benchmark-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Métriques de performance
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Métriques de Performance")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="performance-metrics-table"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Analyse Drawdown
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Analyse des Drawdowns")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="performance-drawdown-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_transactions_layout():
    """
    Crée le layout pour l'analyse des transactions.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Analyse des Transactions"),
                            html.P(
                                "Détail et performance des transactions exécutées",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Filtres
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Période"),
                                                            dcc.DatePickerRange(
                                                                id="transactions-date-range",
                                                                start_date_placeholder_text="Date de début",
                                                                end_date_placeholder_text="Date de fin",
                                                                calendar_orientation="horizontal",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Type de Transaction"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="transactions-type",
                                                                options=[
                                                                    {
                                                                        "label": "Toutes",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Achats",
                                                                        "value": "achat",
                                                                    },
                                                                    {
                                                                        "label": "Ventes",
                                                                        "value": "vente",
                                                                    },
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Actif"),
                                                            dcc.Dropdown(
                                                                id="transactions-asset",
                                                                options=[
                                                                    {
                                                                        "label": "Tous",
                                                                        "value": "all",
                                                                    }
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Résumé des transactions
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Résumé des Transactions")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="transactions-summary"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            # Graphiques des transactions
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Volume des Transactions par Jour")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="transactions-volume-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Performance des Transactions")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="transactions-performance-chart"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Liste détaillée des transactions
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Liste des Transactions")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="transactions-table"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_allocation_layout():
    """
    Crée le layout pour l'analyse de l'allocation d'actifs.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Analyse de l'Allocation d'Actifs"),
                            html.P(
                                "Évolution et optimisation de l'allocation du portefeuille",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Sélection de date
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Label("Date d'allocation"),
                                            dcc.DatePickerSingle(
                                                id="allocation-date",
                                                placeholder="Sélectionner une date",
                                            ),
                                            html.Div(
                                                dbc.Button(
                                                    "Dernière allocation",
                                                    id="btn-latest-allocation",
                                                    color="primary",
                                                    className="mt-2",
                                                ),
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            # Allocation actuelle
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Allocation Actuelle")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="allocation-current-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Détail de l'Allocation")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="allocation-details-table"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Évolution de l'allocation
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Évolution de l'Allocation dans le Temps"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="allocation-evolution-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Comparaison avec l'allocation optimale
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Comparaison avec l'Allocation Optimale"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="allocation-comparison-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Suggestions de Rééquilibrage")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="allocation-rebalancing-suggestions"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


def create_risk_layout():
    """
    Crée le layout pour l'analyse des risques.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Analyse des Risques"),
                            html.P(
                                "Mesures et visualisations des risques du portefeuille",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Métriques de risque
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Métriques de Risque Principales")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="risk-metrics-summary"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Graphiques de risque
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Évolution de la VaR")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="risk-var-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Drawdowns")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="risk-drawdown-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Distribution des Rendements")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="risk-distribution-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Contribution au Risque par Actif")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="risk-contribution-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Scénarios de stress
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Scénarios de Stress")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="risk-stress-scenarios"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_factors_layout():
    """
    Crée le layout pour l'analyse des facteurs.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Analyse des Facteurs"),
                            html.P(
                                "Exposition et contribution des facteurs de risque",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Exposition actuelle
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Exposition aux Facteurs Actuelle")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="factors-exposure-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Évolution des expositions
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Évolution des Expositions aux Facteurs"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(id="factors-evolution-chart"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Contribution des facteurs
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Contribution des Facteurs au Risque")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="factors-risk-contribution-chart"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Contribution des Facteurs au Rendement"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="factors-return-contribution-chart"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Analyse des performances par facteur
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Performance Attribuée aux Facteurs")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="factors-attribution-table"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_viz3d_layout():
    """
    Crée le layout pour les visualisations 3D.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Visualisations 3D"),
                            html.P(
                                "Explorez les relations entre les indicateurs et la performance en 3D",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Sélecteurs de visualisation
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Type de visualisation"),
                                                            dcc.Dropdown(
                                                                id="viz3d-type",
                                                                options=[
                                                                    {
                                                                        "label": "Surface d'indicateurs",
                                                                        "value": "indicator_surface",
                                                                    },
                                                                    {
                                                                        "label": "Trajectoire du portefeuille",
                                                                        "value": "portfolio_trajectory",
                                                                    },
                                                                    {
                                                                        "label": "Clusters de trades",
                                                                        "value": "trade_clusters",
                                                                    },
                                                                ],
                                                                value="indicator_surface",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Période"),
                                                            dcc.DatePickerRange(
                                                                id="viz3d-date-range",
                                                                start_date_placeholder_text="Date de début",
                                                                end_date_placeholder_text="Date de fin",
                                                                calendar_orientation="horizontal",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Paramètres additionnels"),
                                                            dbc.Button(
                                                                "Configurer",
                                                                id="viz3d-config-button",
                                                                color="primary",
                                                                className="mt-2",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Paramètres spécifiques (affichés en fonction de la visualisation sélectionnée)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Collapse(
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                # Paramètres pour surface d'indicateurs
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Axe X"),
                                                                        dcc.Dropdown(
                                                                            id="viz3d-x-axis",
                                                                            options=[],
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Axe Y"),
                                                                        dcc.Dropdown(
                                                                            id="viz3d-y-axis",
                                                                            options=[],
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Axe Z (hauteur)"),
                                                                        dcc.Dropdown(
                                                                            id="viz3d-z-axis",
                                                                            options=[],
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    id="viz3d-indicator-params",
                                                ),
                                                # Paramètres pour trajectoire du portefeuille
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Fenêtre temporelle"),
                                                                        dcc.Slider(
                                                                            id="viz3d-window",
                                                                            min=5,
                                                                            max=60,
                                                                            step=5,
                                                                            value=30,
                                                                            marks={i: str(i) for i in range(5, 61, 10)},
                                                                        ),
                                                                    ],
                                                                    width=12,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    id="viz3d-trajectory-params",
                                                    style={"display": "none"},
                                                ),
                                                # Paramètres pour clusters de trades
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Nombre de clusters"),
                                                                        dcc.Slider(
                                                                            id="viz3d-clusters",
                                                                            min=2,
                                                                            max=10,
                                                                            step=1,
                                                                            value=5,
                                                                            marks={i: str(i) for i in range(2, 11, 1)},
                                                                        ),
                                                                    ],
                                                                    width=12,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    id="viz3d-clusters-params",
                                                    style={"display": "none"},
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                id="viz3d-config-collapse",
                                is_open=False,
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Visualisation 3D
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5(id="viz3d-title")),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="viz3d-loading",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="viz3d-graph",
                                                        figure={},
                                                        style={"height": "700px"},
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Explications et légende
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Comment interpréter")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="viz3d-explanation"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_postmortem_layout():
    """
    Crée le layout pour l'analyse post-mortem des trades.

    Returns:
        Component Layout
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Analyse Post-Mortem des Trades"),
                            html.P(
                                "Analyse détaillée de la performance des transactions passées",
                                className="lead",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Sélecteurs de période et filtreurs
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Période"),
                                                            dcc.DatePickerRange(
                                                                id="postmortem-date-range",
                                                                start_date_placeholder_text="Date de début",
                                                                end_date_placeholder_text="Date de fin",
                                                                calendar_orientation="horizontal",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Type d'analyse"),
                                                            dcc.Dropdown(
                                                                id="postmortem-analysis-type",
                                                                options=[
                                                                    {
                                                                        "label": "Résumé de performance",
                                                                        "value": "performance_summary",
                                                                    },
                                                                    {
                                                                        "label": "Analyse gains/pertes",
                                                                        "value": "win_loss_analysis",
                                                                    },
                                                                    {
                                                                        "label": "Attribution de performance",
                                                                        "value": "trade_attribution",
                                                                    },
                                                                ],
                                                                value="performance_summary",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Filtres supplémentaires
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Filtres"),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Actif"),
                                                            dcc.Dropdown(
                                                                id="postmortem-asset",
                                                                options=[
                                                                    {
                                                                        "label": "Tous",
                                                                        "value": "all",
                                                                    }
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Direction"),
                                                            dcc.Dropdown(
                                                                id="postmortem-direction",
                                                                options=[
                                                                    {
                                                                        "label": "Toutes",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Achat",
                                                                        "value": "buy",
                                                                    },
                                                                    {
                                                                        "label": "Vente",
                                                                        "value": "sell",
                                                                    },
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Résultat"),
                                                            dcc.Dropdown(
                                                                id="postmortem-result",
                                                                options=[
                                                                    {
                                                                        "label": "Tous",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Gains",
                                                                        "value": "win",
                                                                    },
                                                                    {
                                                                        "label": "Pertes",
                                                                        "value": "loss",
                                                                    },
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ]
                    ),
                ]
            ),
            # Statistiques globales
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Statistiques globales")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="postmortem-stats"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Graphique principal
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5(id="postmortem-title")),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="postmortem-loading",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="postmortem-graph",
                                                        figure={},
                                                        style={"height": "700px"},
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Insights et recommandations
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Insights et recommandations")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="postmortem-insights"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
