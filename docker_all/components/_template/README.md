# _template — New component scaffold

Copy this folder to `components/<your_component_name>/` to add a new component to the stack.

## Checklist

1. Copy: `cp -r _template components/<name>` (e.g. `components/zmq_service`).
2. Implement code in `components/<name>/app/`.
3. Edit `Dockerfile`: update COPY path from `components/_template/app` to `components/<name>/app`, set base image and CMD/ENTRYPOINT as needed.
4. Edit `docker-compose.fragment.yml`: change service name from `new_component` to `<name>`, set `dockerfile: components/<name>/Dockerfile`.
5. Add the service to `docker_all/docker-compose.yml` (paste the fragment under `services:`).
6. For ROS2: use same `ROS_DOMAIN_ID`. For ZMQ: add `ports: ["5555:5555"]` and connect to `tcp://<service_name>:5555` from other containers.

See `docker_all/README.md` for the full “Adding a new component” section.
