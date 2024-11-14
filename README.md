# MIOTI_DS_ProyectoFinal

Proyecto final del Master en Data Science de MIOTI


## Install UV

- **Windows**

    ```bash
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

- **MacOS / Linux**

    ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

- **Pip**

    ```bash
        pip install uv
    ```


## Manage Libs

- **Install all Libs**

    ```bash
        uv sync --dev
    ```

- **Add new Libs**

    ```bash
        uv add "lib_name1" "lib_name2" ...
    ```
