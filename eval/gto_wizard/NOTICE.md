The public client at https://github.com/gtowizard-ai/researcher-api-client
documents `POST /hands` and `POST /hands/{hand_id}/act` against
`https://researcher.gtowizard.com` with an `X-API-Key` header. That
repository has no LICENSE file and GitHub reports `license: null`, so this
tree does not vendor its source. Request and response fields used here follow
the public OpenAPI document at https://researcher.gtowizard.com/openapi.json.
The offline fixture path does not call that API. The live smoke runner posts
only ``/hands`` and ``/hands/{id}/act`` with ``X-API-Key`` and does not vendor
that client. It does not call ``/leaderboard``.
