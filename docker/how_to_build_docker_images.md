# Building Docker Images

```bash
docker build -f new_dockerfile -t debug_code:v1 .
```

Where `-f` is the flag to specify the Dockerfile to use, `-t` is used to tag the image with a name (`debug_code`) and version (`v1`), and the `.` at the end indicates that the build context is the current directory.

# Creating container

```bash
docker run --gpus all -itd -v /tss/kinwai/torch_debug:/workspace --name debug_container debug_code:v1
```

# Going inside the container

```bash
docker exec -it debug_container bash
```