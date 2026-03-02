# Remote machine (Orin) setup for pulling from local registry

The **local machine** builds the `remote_zmq_service` image and runs a Docker Registry on port 5000. The **Orin** (remote) pulls the image from your machine and runs the container. Do the following **once** on the Orin.

## 1. Allow HTTP registry on the Orin

The local registry uses HTTP (port 5000). Docker must be told to allow this.

1. SSH to the Orin (or use the same host if testing):
   ```bash
   ssh wheeltec@192.168.31.142
   ```

2. Edit the Docker daemon config:
   ```bash
   sudo nano /etc/docker/daemon.json
   ```

3. Add or merge `insecure-registries` so the file looks like this (use **your local machine’s LAN IP** instead of `192.168.31.100`):
   ```json
   {
     "insecure-registries": ["192.168.31.100:5000"]
   }
   ```
   If the file already has other keys (e.g. `"log-driver"`), add only the `"insecure-registries"` entry and keep the rest.

4. Restart Docker:
   ```bash
   sudo systemctl restart docker
   ```

5. Check that Docker is up:
   ```bash
   docker info | grep -A5 "Insecure Registries"
   ```
   You should see `192.168.31.100:5000` (or your IP).

## 2. Serial device (for the ZMQ service)

If the robot uses a serial port (e.g. `/dev/ttyUSB0`), ensure it is readable:

```bash
sudo chmod a+rw /dev/ttyUSB0
# or add your user to the dialout group: sudo usermod -aG dialout $USER
```

## 3. Run from the local machine

From your **local** machine (where you have the repo and run the script):

1. Start the local registry **once** (if you haven’t already). The container uses `--restart unless-stopped`, so it will come back automatically after a reboot; you don’t need to run this again:
   ```bash
   cd docker_all
   ./scripts/start_local_registry.sh
   ```

2. (Optional) If your machine has multiple IPs, set the one the Orin can reach:
   ```bash
   export REGISTRY_HOST=192.168.31.100   # your PC’s LAN IP
   ```

3. Run the real profile (build, push, then Orin pulls and runs):
   ```bash
   ./scripts/real_up.sh up --build
   ```

The script will build the image for ARM64, push it to `REGISTRY_HOST:5000`, then SSH to the Orin and run `docker pull` and `docker compose up`.

## Troubleshooting

- **Orin cannot pull (connection refused / TLS error)**  
  Confirm `insecure-registries` in `/etc/docker/daemon.json` on the Orin includes `YOUR_LOCAL_IP:5000` and that Docker was restarted. From the Orin, test: `curl -sI http://YOUR_LOCAL_IP:5000/v2/` (should return HTTP/1.1 200 or 401).

- **Local push fails (http: server gave HTTP response to HTTPS client)**  
  The daemon on your **local** machine must allow HTTP to the registry. Edit `/etc/docker/daemon.json` and add:
  `"insecure-registries": ["127.0.0.1:5000", "YOUR_LAN_IP:5000"]` (e.g. `192.168.31.97:5000`). Then run `sudo systemctl restart docker` and `docker start monty_registry`.

- **REGISTRY_HOST not set**  
  The script tries to detect it with `hostname -I`. If you have multiple IPs, set `REGISTRY_HOST` to the IP the Orin uses to reach your machine.
