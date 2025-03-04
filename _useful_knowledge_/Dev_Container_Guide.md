
 # 🚀 Setting Up and Using Dev Containers in WSL2 with Docker

## **📌 1. Overview**
This guide walks you through the complete setup of **WSL2, Docker, VS Code, and Dev Containers**, ensuring a smooth and efficient development workflow.

### **📌 Steps Covered:**
1. 🎯 **Overview** - Understanding dev containers, their pros/cons, and system requirements.
2. 🏗️ **Installing WSL2** - Setting up the Windows Subsystem for Linux.
3. 🐳 **Installing Docker in WSL2** - Ensuring a native, resource-efficient setup.
4. 🛠️ **VS Code Setup** - Integrating with Docker and Dev Containers.
5. 📦 **Building a Dev Container** - Using `devcontainer.json` and `Dockerfile`.
6. ✅ **Verifying the Environment** - Ensuring Rust, Python, R, and `uv` work correctly.
7. ⚙️ **Resource Management** - Handling containers efficiently.
8. 🚀 **Example Workflow** - From booting your PC to shutting down Docker properly.

### **📌 System Requirements:**
 * **Windows 10/11 (2004 or later)**

 * **WSL2 Enabled**

 * **A Linux Distribution (Ubuntu Recommended)**

 * **Docker Installed in WSL2**

 * **VS Code with Remote-WSL and Dev Containers Extensions**

 * **Active Internet Connection** (for installations)

### **📌 Why Use Dev Containers?**

**Pros:**

✅ Consistent development environment across machines.

✅ Avoids dependency conflicts between projects.

✅ Works across Windows/Linux/Mac seamlessly.

✅ Easy to onboard team members with pre-configured environments.

**Cons:**

❌ Requires **more system resources** (CPU/RAM usage).

❌ **Not always needed** for simple projects.

❌ **Storage-heavy** (Docker images can consume GBs of space).

---

## **🏗️ 2. Installing WSL2**
1. Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Restart your computer when prompted.
3. Verify installation:
   ```powershell
   wsl --list --verbose
   ```

---

## **🐳 3. Installing Docker in WSL2**
1. **Update the package index and install prerequisites:**
   ```bash
   sudo apt update && sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
   ```
2. **Create the Keyring Directory and Add Docker’s Official GPG Key:**
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```
3. **Add the Docker Repository:**
   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```
4. **Update the package index again and install Docker:**
   ```bash
   sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io
   ```
5. **Enable and start Docker:**
   ```bash
   sudo service docker start
   ```
6. **Verify Docker installation:**
   ```bash
   docker --version
   ```
7. **Allow user access to Docker without `sudo`:**
   ```bash
   sudo usermod -aG docker $USER
   exec su -l $USER
   ```

---

## **🛠️ 4. VS Code Setup with Dev Containers**
1. Install **Visual Studio Code** from [here](https://code.visualstudio.com/).
2. Install these VS Code extensions:
   - 🖥 **Remote - WSL** (`ms-vscode-remote.remote-wsl`)
   - 📦 **Dev Containers** (`ms-vscode-remote.remote-containers`)
   - 🐳 **Docker** (`ms-azuretools.vscode-docker`)
3. Open VS Code inside WSL:
   ```bash
   code .
   ```

---

## **📦 5. Building a Dev Container**

### **📌 a. Creating the Dev Container Configuration**
1. In your project root, create a `.devcontainer` directory:
   ```bash
   mkdir .devcontainer
   ```
2. Inside `.devcontainer`, create `devcontainer.json` using the attached file contents.
3. Inside `.devcontainer`, create a `Dockerfile` using the attached file contents.

### **📌 b. Rebuilding the Dev Container**
1. Open **Command Palette** (`Ctrl+Shift+P`).
2. Select **"Dev Containers: Rebuild and Reopen in Container"**.

---

## **✅ 6. Verifying That Everything Works**
After opening the dev container terminal, check:
```bash
python --version
rustc --version
cargo --version
uv --version
which uv
```

---

## **⚙️ 7. Resource Management & Turning Docker On/Off**
- **Start Docker manually:**
  ```bash
  sudo service docker start
  ```
- **Stop Docker when done:**
  ```bash
  sudo service docker stop
  ```
- **Check running containers:**
  ```bash
  docker ps
  ```
- **Remove unused containers/images to free space:**
  ```bash
  docker system prune -a
  ```

---

## **🚀 8. Example Workflow Using a Dev Container**

### **📌 a. Starting Your Development Session (After Booting Your PC)**
1. Open **PowerShell** and start WSL:
   ```powershell
   wsl
   ```
2. Start Docker manually if it’s not running:
   ```bash
   sudo service docker start
   ```
3. Navigate to your project:
   ```bash
   cd ~/my-project
   ```
4. Open VS Code:
   ```bash
   code .
   ```
5. Reopen in Dev Container:
   - Press `Ctrl+Shift+P`
   - Select **"Reopen in Container"**

---

## **🎯 Conclusion**
Following this guide, you now have a fully functional Dev Container with **Python, Rust, R, and uv**. This setup ensures a reproducible, efficient, and streamlined development experience! 🚀