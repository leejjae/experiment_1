#!/usr/bin/env bash
# WebArena 사이트를 "호스트"에서 기동한다 (개발 컨테이너 안이 아니라).
#
#   호스트 셸에서:  bash setup_host_sites.sh pull && bash setup_host_sites.sh up
#
# 기동된 사이트는 개발 컨테이너(172.17.0.3)에서 http://172.17.0.1:<port> 로 접속된다.
# 이미지는 Docker Hub에 이미 빌드되어 있으므로 tar 다운로드/docker load 는 필요 없다.

set -euo pipefail

# 개발 컨테이너에서 호스트를 가리키는 주소. 도커 기본 브리지의 게이트웨이.
# 컨테이너 안에서 `python3 -c "import struct,socket;[print(socket.inet_ntoa(struct.pack('<L',int(l.split()[2],16)))) for l in open('/proc/net/route').readlines()[1:] if l.split()[1]=='00000000']"`
# 로 확인한 값과 일치해야 한다.
HOST_ADDR="${HOST_ADDR:-172.17.0.1}"

IMG_SHOPPING="webarenaimages/shopping_final_0712"
IMG_SHOPPING_ADMIN="webarenaimages/shopping_admin_final_0719"
IMG_GITLAB="webarenaimages/gitlab-populated-final"
IMG_FORUM="webarenaimages/postmill-populated-exposed-withimg"

cmd="${1:-help}"

case "$cmd" in

precheck)
  echo "== docker =="; docker version --format '{{.Server.Version}}'
  echo "== 도커 데이터 루트 여유 공간 (>=200GB 필요) =="
  root=$(docker info --format '{{.DockerRootDir}}')
  echo "root=$root"; df -h "$root" | tail -1
  echo "== 포트 충돌 확인 =="
  for p in 7770 7780 8023 9999; do
    if ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN; then echo "  !! $p 사용 중"; else echo "  $p 비어있음"; fi
  done
  ;;

pull)
  # 압축 기준 합계 ~142GB. 네트워크 상태에 따라 수 시간 걸린다.
  for img in "$IMG_SHOPPING" "$IMG_SHOPPING_ADMIN" "$IMG_GITLAB" "$IMG_FORUM"; do
    echo "== pull $img"; docker pull "$img"
  done
  ;;

up)
  docker rm -f shopping shopping_admin gitlab forum 2>/dev/null || true

  # --shm-size 필수: 도커 기본 /dev/shm 은 64MB 다. GitLab 은 Prometheus 메트릭을
  # 메모리 맵 파일로 쓰는데, 64MB 를 넘기는 순간 모든 계측이 `IOError: unmapped file`
  # 로 터지고 그 계측이 미들웨어 체인에 있어서 요청 자체가 500/502 가 된다.
  # 부팅 직후엔 멀쩡하다가 3~4분에 걸쳐 전면적으로 무너지는 증상으로 나타난다.
  docker run --name shopping       -p 7770:80   --shm-size=2g -d "$IMG_SHOPPING"
  docker run --name shopping_admin -p 7780:80   --shm-size=2g -d "$IMG_SHOPPING_ADMIN"
  docker run --name forum          -p 9999:80   --shm-size=1g -d "$IMG_FORUM"
  # gitlab 은 entrypoint 를 명시해야 한다. 부팅에 5분 이상 걸린다.
  docker run --name gitlab -d -p 8023:8023 --shm-size=2g "$IMG_GITLAB" /opt/gitlab/embedded/bin/runsvdir-start

  echo "서비스 기동 대기 (Magento ~60s)..."; sleep 75

  # --- base URL 재작성 ---------------------------------------------------
  # Magento/GitLab 은 절대 URL 로 리다이렉트하므로, 개발 컨테이너의 브라우저가
  # 실제로 접근하는 주소($HOST_ADDR)와 일치시켜야 한다. localhost 로 두면
  # 로그인 직후 컨테이너가 자기 자신에게 붙으려다 실패한다.
  for pair in "shopping:7770" "shopping_admin:7780"; do
    name="${pair%%:*}"; port="${pair##*:}"
    echo "== $name base_url -> http://$HOST_ADDR:$port"
    docker exec "$name" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$HOST_ADDR:$port"
    docker exec "$name" mysql -u magentouser -pMyPassword magentodb \
      -e "UPDATE core_config_data SET value='http://$HOST_ADDR:$port/' WHERE path = 'web/secure/base_url';"
    docker exec "$name" /var/www/magento2/bin/magento cache:flush
  done

  echo "gitlab 부팅 대기 (최대 5분)..."; sleep 300
  docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$HOST_ADDR:8023'|" /etc/gitlab/gitlab.rb
  docker exec gitlab gitlab-ctl reconfigure

  echo; echo "완료. 개발 컨테이너에서 아래로 확인:"
  echo "  bash setup_host_sites.sh verify"
  ;;

regen-gitlab)
  # GitLab 만 깨끗하게 다시 만든다. 이미지에 데이터가 구워져 있고 볼륨을 안 쓰므로
  # 컨테이너를 지워도 데이터는 그대로다 (오히려 초기 상태로 돌아간다).
  # 이미 pull 해둔 이미지를 쓰므로 재다운로드도 없다.
  docker rm -f gitlab 2>/dev/null || true
  docker run --name gitlab -d -p 8023:8023 --shm-size=2g \
    "$IMG_GITLAB" /opt/gitlab/embedded/bin/runsvdir-start

  echo "gitlab 부팅 대기 (5분)..."; sleep 300
  docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$HOST_ADDR:8023'|" /etc/gitlab/gitlab.rb
  docker exec gitlab gitlab-ctl reconfigure
  echo "reconfigure 후 안정화 대기 (3분)..."; sleep 180
  docker exec gitlab gitlab-ctl status
  echo; echo "완료. /dev/shm 확인:"; docker exec gitlab df -h /dev/shm
  ;;

verify)
  # 개발 컨테이너 안에서 실행해도 된다 (HOST_ADDR 로 접근하므로).
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
