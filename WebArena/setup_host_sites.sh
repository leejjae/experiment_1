#!/usr/bin/env bash
# Start the WebArena sites on the host (not inside the development container).
#
#   On the host shell:  bash setup_host_sites.sh pull && bash setup_host_sites.sh up
#
# The development container (172.17.0.3) reaches the sites at http://172.17.0.1:<port>.
# The images are prebuilt on Docker Hub, so no tar download or docker load is needed.

set -euo pipefail

# Address of the host as seen from the development container: the default Docker bridge gateway.
# It must match the value printed inside the container by `python3 -c "import struct,socket;[print(socket.inet_ntoa(struct.pack('<L',int(l.split()[2],16)))) for l in open('/proc/net/route').readlines()[1:] if l.split()[1]=='00000000']"`
# (the default route gateway).
HOST_ADDR="${HOST_ADDR:-172.17.0.1}"

IMG_SHOPPING="webarenaimages/shopping_final_0712"
IMG_SHOPPING_ADMIN="webarenaimages/shopping_admin_final_0719"
IMG_GITLAB="webarenaimages/gitlab-populated-final"
IMG_FORUM="webarenaimages/postmill-populated-exposed-withimg"

cmd="${1:-help}"

case "$cmd" in

precheck)
  echo "== docker =="; docker version --format '{{.Server.Version}}'
  echo "== Free space in the Docker data root (>= 200 GB needed) =="
  root=$(docker info --format '{{.DockerRootDir}}')
  echo "root=$root"; df -h "$root" | tail -1
  echo "== Checking for port conflicts =="
  for p in 7770 7780 8023 9999; do
    if ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN; then echo "  !! $p in use"; else echo "  $p free"; fi
  done
  ;;

pull)
  # About 142 GB compressed in total; takes hours depending on the network.
  for img in "$IMG_SHOPPING" "$IMG_SHOPPING_ADMIN" "$IMG_GITLAB" "$IMG_FORUM"; do
    echo "== pull $img"; docker pull "$img"
  done
  ;;

up)
  docker rm -f shopping shopping_admin gitlab forum 2>/dev/null || true

  # --shm-size is required: Docker's default /dev/shm is 64 MB. GitLab keeps Prometheus metrics
  # in memory-mapped files; once they exceed 64 MB every metric raises `IOError: unmapped file`,
  # and because the metrics sit in the middleware chain, requests themselves fail with 500/502.
  # Symptom: fine right after boot, then a complete breakdown over 3-4 minutes.
  docker run --name shopping       -p 7770:80   --shm-size=2g -d "$IMG_SHOPPING"
  docker run --name shopping_admin -p 7780:80   --shm-size=2g -d "$IMG_SHOPPING_ADMIN"
  docker run --name forum          -p 9999:80   --shm-size=1g -d "$IMG_FORUM"
  # GitLab needs an explicit entrypoint and takes more than 5 minutes to boot.
  docker run --name gitlab -d -p 8023:8023 --shm-size=2g "$IMG_GITLAB" /opt/gitlab/embedded/bin/runsvdir-start

  echo "Waiting for services to start (Magento ~60 s)..."; sleep 75

  # --- rewrite base URLs ------------------------------------------------
  # Magento and GitLab redirect to absolute URLs, so the base URL must match the address
  # the browser in the development container actually uses ($HOST_ADDR). With localhost,
  # the container tries to connect to itself right after login and fails.
  for pair in "shopping:7770" "shopping_admin:7780"; do
    name="${pair%%:*}"; port="${pair##*:}"
    echo "== $name base_url -> http://$HOST_ADDR:$port"
    docker exec "$name" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$HOST_ADDR:$port"
    docker exec "$name" mysql -u magentouser -pMyPassword magentodb \
      -e "UPDATE core_config_data SET value='http://$HOST_ADDR:$port/' WHERE path = 'web/secure/base_url';"
    docker exec "$name" /var/www/magento2/bin/magento cache:flush
  done

  echo "Waiting for GitLab to boot (up to 5 min)..."; sleep 300
  docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$HOST_ADDR:8023'|" /etc/gitlab/gitlab.rb
  docker exec gitlab gitlab-ctl reconfigure

  echo; echo "Done. Check from the development container:"
  echo "  bash setup_host_sites.sh verify"
  ;;

regen-gitlab)
  # Recreate only GitLab. The data is baked into the image and no volume is used,
  # so removing the container keeps the data (it returns to the initial state).
  # The already-pulled image is reused, so nothing is downloaded again.
  docker rm -f gitlab 2>/dev/null || true
  docker run --name gitlab -d -p 8023:8023 --shm-size=2g \
    "$IMG_GITLAB" /opt/gitlab/embedded/bin/runsvdir-start

  echo "Waiting for GitLab to boot (5 min)..."; sleep 300
  docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$HOST_ADDR:8023'|" /etc/gitlab/gitlab.rb
  docker exec gitlab gitlab-ctl reconfigure
  echo "Waiting for GitLab to settle after reconfigure (3 min)..."; sleep 180
  docker exec gitlab gitlab-ctl status
  echo; echo "Done. /dev/shm:"; docker exec gitlab df -h /dev/shm
  ;;

verify)
  # Can also be run inside the development container (it goes through HOST_ADDR).
  for pair in "shopping:7770:/" "shopping_admin:7780:/admin" "forum:9999:/" "gitlab:8023:/explore"; do
    name="${pair%%:*}"; rest="${pair#*:}"; port="${rest%%:*}"; path="${rest##*:}"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "http://$HOST_ADDR:$port$path" || echo 000)
    printf "  %-16s http://%s:%s%-8s -> %s\n" "$name" "$HOST_ADDR" "$port" "$path" "$code"
  done
  ;;

down)
  docker rm -f shopping shopping_admin gitlab forum 2>/dev/null || true
  ;;

*)
  echo "usage: bash $0 {precheck|pull|up|regen-gitlab|verify|down}"
  exit 1
  ;;
esac
