# Electric arc furnace energy estimate

This project estimates the electrical energy used in one heat of the DSP-120 electric arc furnace (EAF).
LightGBM uses the charge data and the process data for the heat.
Use the result to compare completed heats.
Some inputs, such as the final temperature and total arc time, are available late in the heat.

## Prepare the data

1. Open PowerShell in the project folder.
2. Download the [Kaggle EAF dataset](https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace).
3. Extract the raw CSV files into `data/`.
4. Install `uv` if necessary.
5. Install Python 3.13 if necessary.
6. Install the project dependencies.

   ```powershell
   uv sync --locked
   ```

7. Run the data preparation script.

   ```powershell
   uv run python prepare_data.py
   ```

The script groups the basket charges, material additions, transformer records, temperature records, and gas and oxygen records by heat ID.
It calculates charge totals, arc times, tap mass, and a statistical energy estimate.
It sums the transformer column `MW` for each heat.
The project assumes that this sum is in MWh per heat.
The source does not confirm this unit.

The script writes 20,427 heats to `data/prepared/df_combined.csv`.
It applies the notebook cleaning rules and writes 19,254 heats to `data/prepared/df_combined_cleaned.csv`.
The rules exclude one bad heat ID and filter outliers in six columns.
The model uses the resulting range of operating conditions.

## Review the notebooks

| Notebook | What it shows |
| --- | --- |
| [1. Data preparation](1_data_preparation.ipynb) | Combine the raw records and calculate heat-level inputs and the statistical estimate. |
| [2. Exploratory data analysis](2_eda.ipynb) | Examine the energy, charge, gas, oxygen, and timing data. |
| [3. Data cleaning](3_data_cleaning.ipynb) | Apply the outlier rules and save the data for training. |
| [4. Modelling](4_modelling.ipynb) | Compare Ridge, gradient boosting, and LightGBM with the statistical estimate. See the feature-importance analysis. |
| [Final chemistry](final_chem.ipynb) | Explore final steel chemistry as a separate analysis. |

See the [data description](data_description.md) for the source columns.
Run `prepare_data.py` before you run notebooks 2 or 4.

## Train the model

Run the training script after data preparation.

```powershell
uv run python train.py
```

The script reads `data/prepared/df_combined_cleaned.csv`.
It saves `artifacts/lightgbm_model.txt` and `artifacts/model_metadata.json`.
It uses a shuffled split with seed 42.
There are 11,552 training heats, 3,851 validation heats, and 3,851 test heats.
It chooses the LightGBM settings with the lowest validation RMSE.
It selects 31 leaves and 1,000 trees.

## Model results

MAE is mean absolute error.
RMSE is root mean square error.
R² is the coefficient of determination.
MAE and RMSE use the assumed unit MWh per heat.

### Notebook validation results

These results come from a fresh run of the model cells in [the modelling notebook](4_modelling.ipynb) with the regenerated CSV.
The notebook can show results saved from an earlier run.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Statistical estimate | 7.545 | 8.184 | -8.717 |
| Ridge regression | 1.115 | 1.507 | 0.671 |
| Gradient boosting | 1.008 | 1.394 | 0.718 |
| LightGBM | 0.971 | 1.343 | 0.738 |

The statistical comparison uses `stat_MW × Calc_tap_mass / 1000` to convert kWh/t to MWh per heat.
The calculation uses an estimate of the tapped steel mass.

### Training script test results

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Training median | 1.970 | 2.575 | -0.002 |
| Statistical estimate | 7.517 | 8.145 | -9.021 |
| Ridge regression | 1.100 | 1.416 | 0.697 |
| LightGBM | **0.958** | **1.237** | **0.769** |

The test scores apply to complete records from the same operating mix.
A partial request uses default values. Its error can differ from the reported test error.

## Run the API

### Start the service

Run these commands after training.

```powershell
docker build --platform linux/amd64 --tag eaf-energy:local .
docker run --rm --publish 8080:8080 eaf-energy:local
```

To run the API in the project environment, use this command.

```powershell
uv run uvicorn predict:app --host 127.0.0.1 --port 8080
```

### Check the service

Use `GET /test` to check the model files and make a test prediction.
The endpoint returns HTTP 200 and `{"status":"ready"}` when the service is ready.
It returns HTTP 503 if the model files are missing or cannot be used.

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/test"
```

### Make a prediction

Send one heat as a JSON object to `POST /predict`.
Use the column names from `df_combined_cleaned.csv`.
Use numeric values for the inputs.
Use kilograms for material amounts, degrees Celsius for `TEMP`, and minutes for `power_on` and `power_off`.
The API accepts `HEATID`, `MW`, and the saved row index as metadata.
The model uses the 69 predictor columns.

Omit an input to use its training median.
The default is zero for a material amount if its median is below 1,000 kg.
The response lists the fields that use a default.
The API returns HTTP 422 for an unknown field or an invalid predictor value.

```powershell
$Result = Invoke-RestMethod -Uri "http://localhost:8080/predict" -Method Post -ContentType "application/json" -Body '{"TEMP":1650,"power_on":30}'
$Result | Select-Object energy_mwh, unit, @{Name="defaulted_count"; Expression={$_.defaulted_fields.Count}}
```

The example gives an estimate of 47.82 MWh per heat.
It uses defaults for 67 fields.

## Deploy the Docker image to Google Cloud Run

Use a Google Cloud project with billing enabled.
Install Docker.
Install the Google Cloud CLI.
Use an account with permission to enable APIs, create the registry repository, push the image, and deploy the service.

1. Complete [Train the model](#train-the-model).
   Make sure that `artifacts/` contains both model files.
   The Dockerfile sets the API to listen on `0.0.0.0:8080`.

2. Set the deployment parameters in PowerShell.
   Replace `YOUR_PROJECT_ID` with your Google Cloud project ID.

   ```powershell
   $ProjectId = "YOUR_PROJECT_ID"
   $Region = "europe-west1"
   $Repository = "eaf-models"
   $Service = "eaf-energy"
   $RegistryHost = "$Region-docker.pkg.dev"
   $Image = "$RegistryHost/$ProjectId/$Repository/${Service}:v1"
   ```

3. Sign in to Google Cloud.
   Enable the Cloud Run and Artifact Registry APIs.

   ```powershell
   gcloud auth login
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project $ProjectId
   ```

4. Create the Docker repository in Artifact Registry.
   Run the repository command once per project and region.
   Configure Docker to sign in to Artifact Registry.

   ```powershell
   gcloud artifacts repositories create $Repository --repository-format docker --location $Region --project $ProjectId
   gcloud auth configure-docker $RegistryHost
   ```

5. Build the Docker image for Linux.
   Push the image to Artifact Registry.

   ```powershell
   docker build --platform linux/amd64 --tag $Image .
   docker push $Image
   ```

6. Deploy the image to Cloud Run.
   The service requires authentication.

   ```powershell
   gcloud run deploy $Service --image $Image --region $Region --project $ProjectId --port 8080 --no-allow-unauthenticated
   ```

7. Get the service URL.
   Use an account with the Cloud Run Invoker role to send the two test requests.
   Check that `/test` returns HTTP 200.
   Check that `/predict` returns a prediction.

   ```powershell
   $ServiceUrl = gcloud run services describe $Service --region $Region --project $ProjectId --format 'value(status.url)'
   $Headers = @{ Authorization = "Bearer $(gcloud auth print-identity-token)" }
   (Invoke-WebRequest -Uri "$ServiceUrl/test" -Headers $Headers -UseBasicParsing).StatusCode
   Invoke-RestMethod -Uri "$ServiceUrl/predict" -Method Post -Headers $Headers -ContentType "application/json" -Body '{"TEMP":1650,"power_on":30}'
   ```

[Cloud Run deployment guide](https://cloud.google.com/run/docs/deploying) · [Artifact Registry Docker guide](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)
