Below is the updated Dev Container Guide in Markdown. I’ve preserved the original content while adding clarifications, additional troubleshooting tips, and updates based on our latest discussions.

---

# 🚀 Dev Container Guide for WSL2 with Docker and VS Code

This guide walks you through setting up and using development containers on WSL2 with Docker and Visual Studio Code. It covers installing the necessary components, configuring your dev container with a robust and flexible Dockerfile and devcontainer.json, and includes tips for GPU usage, mounting host directories, and handling Python dependencies via requirements.txt.

---

## **1. Overview**

This guide covers:
- An introduction to dev containers and their benefits.
- Installing and configuring WSL2.
- Installing Docker inside WSL2.
- Setting up Visual Studio Code with Remote – WSL and Dev Containers extensions.
- Installing the Dev Container CLI.
- Creating a dev container using a `devcontainer.json` and a Dockerfile.
- Conditional installation of Python dependencies (via requirements.txt).
- Tips for GPU support and mounting Windows directories.
- Managing resources and troubleshooting common issues.
- A step-by-step example workflow.

**System Requirements:**
- Windows 10/11 (build 2004 or later)
- WSL2 enabled with a Linux distro (Ubuntu recommended for GPU work, Alpine for lightweight setups, etc.)
- Docker installed in WSL2 (or Docker Desktop with WSL integration)
- Visual Studio Code with:
  - Remote – WSL (`ms-vscode-remote.remote-wsl`)
  - Dev Containers (`ms-vscode-remote.remote-containers`)
  - Docker (`ms-azuretools.vscode-docker`)
- Active Internet connection

---

## **2. Installing WSL2**

1. **Open PowerShell as Administrator** and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Restart your computer if prompted.
3. Verify the installation:
   ```powershell
   wsl --list --verbose
   ```

---

## **3. Installing Docker in WSL2**

1. **Update packages and install prerequisites:**
   ```bash
   sudo apt update && sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
   ```
2. **Add Docker’s official GPG key and repository:**
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```
3. **Install Docker:**
   ```bash
   sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io
   ```
4. **Start Docker:**
   ```bash
   sudo service docker start
   ```
5. **Verify Docker:**
   ```bash
   docker --version
   ```
6. **Allow user access (optional):**
   ```bash
   sudo usermod -aG docker $USER
   exec su -l $USER
   ```

---

## **4. VS Code Setup with Dev Containers**

1. **Install Visual Studio Code** from [here](https://code.visualstudio.com/).
2. **Install these VS Code extensions:**
   - Remote – WSL (`ms-vscode-remote.remote-wsl`)
   - Dev Containers (`ms-vscode-remote.remote-containers`)
   - Docker (`ms-azuretools.vscode-docker`)
   - Plus any language-specific or productivity extensions you use.
3. **Open VS Code inside WSL:**  
   From your WSL terminal, run:
   ```bash
   code .
   ```

---

## **5. Installing the Dev Container CLI**

To enable advanced command-line operations with dev containers:

1. **Open your WSL terminal and install via npm:**
   ```bash
   npm install -g @devcontainers/cli
   ```
2. **Verify the installation:**
   ```bash
   devcontainer --version
   ```
   Ensure that the CLI is available in your PATH.

---

## **6. Creating a Dev Container**

### **a. Project Structure & Configuration**

Place your dev container configuration in a folder named `.devcontainer` at the root of your project. For example, your project might look like:

```
my-project/
  ├─ .devcontainer/
  │    ├─ Dockerfile
  │    └─ devcontainer.json
  ├─ requirements.txt
  └─ <other project files>
```

### **b. Dockerfile**

Below is a sample Dockerfile using an Alpine base image that meets the following criteria:
- Uses Alpine Linux as the base image.
- Installs Python3, pip, and other minimal packages.
- Creates and switches to a non-root user (`vscode`).
- Installs uv via a curl command.
- Implements conditional logic for installing dependencies from a requirements.txt file.

```dockerfile
# Use Alpine Linux as the base image
ARG BASE_IMAGE=alpine:latest
FROM ${BASE_IMAGE}

# Install essential packages: Python3, pip, bash, curl, and ca-certificates
RUN apk update && apk add --no-cache \
    python3 \
    py3-pip \
    bash \
    curl \
    ca-certificates

# Create a non-root user 'vscode'
RUN adduser -D vscode

# Set working directory for vscode user
WORKDIR /home/vscode

# Switch to non-root user
USER vscode

# Install uv (this installs uv into /home/vscode/.local/bin)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/home/vscode/.local/bin:$PATH"

# Copy requirements.txt from the build context (even if empty, to prevent COPY errors)
COPY requirements.txt ./

# Conditionally install Python dependencies:
# If requirements.txt exists and is non-empty, use uv pip install with --system.
RUN if [ -f requirements.txt ] && [ -s requirements.txt ]; then \
      uv pip install --system -r requirements.txt; \
    else \
      echo "No requirements to install"; \
    fi

# Expose a port if your container runs a service (optional)
# EXPOSE 3000

# Default command
CMD ["bash"]
```

### **c. devcontainer.json**

Below is an updated devcontainer.json that specifies the Dockerfile, sets VS Code settings, binds the Docker socket (if needed), and ensures that the container runs as the `vscode` user.

```json
{
  "name": "Wisp",
  "build": {
    "dockerfile": "Dockerfile",
    "context": "..",
    "args": {
      "BASE_IMAGE": "alpine:latest"
    }
  },
  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.shell.linux": "/bin/bash"
      },
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-vscode-remote.remote-containers",
        "ms-azuretools.vscode-docker",
        "ms-toolsai.jupyter",
        "xelad0m.jupyter-toc",
        "catppuccin.catppuccin-vsc",
        "ms-toolsai.datawrangler",
        "fill-labs.dependi",
        "djsaunders1997.uv-wingman",
        "visualstudioexptteam.vscodeintellicode",
        "charliermarsh.ruff",
        "ms-vscode.live-server"
      ]
    }
  },
  "runArgs": [
    "-v", "/var/run/docker.sock:/var/run/docker.sock"
  ],
  "remoteUser": "vscode"
}
```

---

## **7. Verifying Your Environment**

After opening your dev container, verify that the essential tools are working by running in the container terminal:

```bash
python --version
which uv
uv --version
```

If you have a requirements.txt, ensure that its packages are installed. If you run into errors with uv pip install, note that uv may require a virtual environment or the `--system` flag (as used above) to work correctly.

> **Note:**  
> In earlier iterations, issues arose with the requirements.txt logic—if the file was empty or not in the expected state, uv would error out. The conditional check in the Dockerfile ensures that uv only attempts installation if the file is present and non-empty. If you ever face issues, you might need to adjust the logic or comment it out as a workaround.

---

## **8. Resource Management & Persistence**

- **Image Caching:**  
  Docker caches each layer of your image. Base images, installed packages, and dependencies are only re-downloaded if there are changes or if you clear your cache.
  
- **Container Persistence:**  
  - Containers built from your image retain data in volumes or bind mounts, but they do not re-download the image every time they start.
  - After a `wsl --shutdown`, your container is stopped, but the image and any persistent volumes remain on disk. No extra internet data is used unless you explicitly pull new layers or updates.
  
- **Cleaning Up:**  
  Use commands like `docker system prune -a` to clean up unused images and containers, but be cautious not to remove what you need.

---

## **9. Example Workflow**

A typical day might involve:

1. **Booting your PC.**
2. **Starting WSL:**  
   ```powershell
   wsl
   ```
3. **Starting Docker (if needed):**  
   ```bash
   sudo service docker start
   ```
4. **Navigating to your project directory:**  
   ```bash
   cd /home/<your-user>/path/to/project
   ```
5. **Opening the project in VS Code with the dev container:**  
   From Windows CMD:
   ```batch
   wsl bash -c "sudo service docker start && devcontainer up --workspace-folder /home/<your-user>/path/to/project && code --remote dev-container+$(wslpath -w /home/<your-user>/path/to/project)"
   ```
   > **Tip:** If the container is already running, you can simply use the VS Code command; otherwise, `devcontainer up` ensures the container is built and started.
6. **Coding, running tests, and using source control.**
7. **Stopping containers and cleaning up when done.**

---

## **10. Additional Tips & Troubleshooting**

- **GPU Support:**  
  Alpine Linux doesn’t have native NVIDIA support out-of-the-box. If you need GPU acceleration, consider using an official NVIDIA CUDA base image (often based on Ubuntu) or be prepared to install additional drivers/libraries manually.
  
- **Mounting Host Directories:**  
  You can bind mount Windows directories (e.g., your C:\ drive) by adding a runArg in your devcontainer.json:
  ```json
  "runArgs": [
    "-v", "/mnt/c/Users/cml_p:/mnt/hostusers"
  ]
  ```
  This gives you access to any file under that directory inside your container.
  
- **Docker CLI Inside Container:**  
  If you need to run Docker commands from within your container, install the Docker CLI (e.g., using `apk add --no-cache docker-cli`) and bind mount the Docker socket.
  
- **VS Code “Remote” Warning:**  
  The message “Option 'remote' is defined more than once” appears when multiple remote options are provided (for example, from both the command line and the container configuration). In practice, VS Code picks one value. If everything functions as expected, this warning is harmless.
  
- **Requirements.txt Logic:**  
  We added conditional logic to only install packages if `requirements.txt` exists and is non-empty. This was necessary because an empty or misconfigured requirements.txt can cause uv to fail. If you run into issues, ensure that your requirements file is correctly formatted, or adjust the logic accordingly.

---

## **11. Conclusion**

By following this guide, you have a robust, reproducible dev container environment for your development needs. The configuration is flexible, allowing conditional dependency installation, host file access, and integration with Docker and VS Code. Whether you’re working on Python, Rust, or any other technology, this setup should provide a consistent and efficient development workflow.

Happy coding, and feel free to adjust and extend this guide as your needs evolve!

---

If you have any further questions or need additional updates, let me know!
