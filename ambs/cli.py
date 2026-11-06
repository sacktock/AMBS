import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
train_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(train_app, name="train")

_EXTRA = {"allow_extra_args": True, "ignore_unknown_options": True}


def _train(module: str, ctx: typer.Context) -> None:
    import importlib

    importlib.import_module(module).main(ctx.args)


@train_app.command("dreamerv3", context_settings=_EXTRA)
def dreamerv3(ctx: typer.Context) -> None:
    """Train DreamerV3."""
    _train("ambs.dreamerv3.train", ctx)


@train_app.command("dreamerv3-lag", context_settings=_EXTRA)
def dreamerv3_lag(ctx: typer.Context) -> None:
    """Train Lagrangian DreamerV3."""
    _train("ambs.dreamerv3_lag.train", ctx)


@train_app.command("dreamerv3-shield", context_settings=_EXTRA)
def dreamerv3_shield(ctx: typer.Context) -> None:
    """Train shielded DreamerV3."""
    _train("ambs.dreamerv3_shield.train", ctx)
