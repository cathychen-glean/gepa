# GEPA core for Cortex

Installable distribution: `glean-gepa-core==0.1.0`. Python import namespace:
`gepa`. This independent package contains only the 18 Python modules listed in
Cortex's `python_scio/cortex/impl/gepa/vendor/BUILD.bazel`, plus package metadata
and the MIT license. The BUILD file is not part of the Python distribution.

The source snapshot was copied from the Scio checkout at commit
`d006f7d5e1a7cb5fc5c913a6ea294ea56442c569`, directory
`python_scio/cortex/impl/gepa/vendor/gepa`. The Python files are preserved without
modification, including the minimal package initializers. This snapshot is
independent of the full GEPA project in the parent directory and of upstream tags.

## Build and install locally

From the root of the GEPA repository:

```bash
uv build --no-sources packages/glean-gepa-core --out-dir packages/glean-gepa-core/dist
uv run --no-project --python 3.12 \
  --with ./packages/glean-gepa-core/dist/glean_gepa_core-0.1.0-py3-none-any.whl \
  python -I -c 'from gepa.core.engine import GEPAEngine; print(GEPAEngine)'
```

There are no mandatory third-party runtime dependencies. Optional checkpoint
serialization can use `glean-gepa-core[checkpoint]` for cloudpickle; W&B and
MLflow logging require their corresponding extras. Without cloudpickle the
existing code falls back to standard pickle, which cannot serialize every object.
The optional integrations are not included in the base package smoke checks.

## Publish for Cortex

Scio's MODULE.bazel already resolves packages through:

`https://us-central1-python.pkg.dev/scio-engineering/glean-pip/simple`

Scio's registry tooling identifies this as a virtual download index. Obtain the
name of a **standard Python repository backing that index** from the team that
maintains it. Publishing to an arbitrary repository does not make the package
visible through `glean-pip`.

Once the team provides its upload URL and you have write access, publish just
these artifacts (replace the placeholder):

```bash
UV_PUBLISH_USERNAME=oauth2accesstoken \
UV_PUBLISH_PASSWORD="$(gcloud auth print-access-token)" \
uv publish --publish-url 'https://REGION-python.pkg.dev/PROJECT/STANDARD_REPOSITORY/' \
  packages/glean-gepa-core/dist/glean_gepa_core-0.1.0-py3-none-any.whl \
  packages/glean-gepa-core/dist/glean_gepa_core-0.1.0.tar.gz
```

Do not upload credentials into Git. The command obtains a short-lived token
from your existing gcloud login. This package has not been published yet.
Use a new version for subsequent releases, and record the source commit for each.

## Add to Cortex

After the package is available through `glean-pip`:

1. In Scio's `python_scio/requirements/bazel_requirements.in`, add:

   ```text
   glean-gepa-core==0.1.0
   ```

2. From the Scio repository root, regenerate the platform lockfiles and manifest
   with its existing helper:

   ```bash
   bash python_scio/requirements/update_bazel_requirements.sh
   ```

   Commit the resulting lockfile and manifest changes with the requirement.

3. On the Cortex `py_library` or `py_binary` target that imports GEPA, add the
   dependency `@pip//glean_gepa_core`. Replace any dependency on
   `//python_scio/cortex/impl/gepa/vendor:gepa` in the same runtime dependency
   graph. For example, the relevant portion of a BUILD file is:

   ```starlark
   deps = [
       "@pip//glean_gepa_core",
   ],
   ```

4. Keep the Python imports as:

   ```python
   from gepa.core.adapter import GEPAAdapter
   from gepa.core.engine import GEPAEngine
   from gepa.core.result import GEPAResult
   ```

   This deliberately does not expose the full GEPA API (`from gepa import
   optimize` is not supported). Do not install public `gepa` or retain the vendor
   target in the same runtime: all provide the same `gepa` namespace and could
   shadow one another. After updating consumers, the old vendor files can be
   removed from Scio.

5. Build and run the affected Cortex tests in Scio. Local wheel smoke checks
   verify the package itself; they do not replace service integration tests.

The Scio checkout has not been modified by this packaging task.
