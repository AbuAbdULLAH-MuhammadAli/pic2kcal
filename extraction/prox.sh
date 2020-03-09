trap 'kill -HUP 0' EXIT
watchport=50030
listenport=17300
for server in 165.227.39.5 165.227.45.215 159.89.41.126 159.89.41.114 138.197.222.205 165.227.12.53 188.166.14.177 188.166.3.88 159.65.10.123 165.227.137.239 165.227.225.146 46.101.48.190; do

        autossh -4 -M $watchport -TN -D $listenport -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 root@$server &
        ((watchport += 2))
        ((listenport += 1))
done

wait
