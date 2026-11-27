"""HTTP routers, one module per resource. All paths live under /api."""

from lab.api import datasets, gt, media, models, preview, runs, score, system

ROUTERS = [system.router, media.router, datasets.router, gt.router, score.router, models.router, preview.router,
           runs.router]
