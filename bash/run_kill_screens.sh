#!/bin/bash

# Define the list of servers to connect to
servers=("kiev.ee.ucl.ac.uk" "lyon.ee.ucl.ac.uk" "verona.ee.ucl.ac.uk" "riga.ee.ucl.ac.uk" "oslo.ee.ucl.ac.uk" "monaco.ee.ucl.ac.uk" "chester.ee.ucl.ac.uk" "budapest.ee.ucl.ac.uk")

# Define the SSH username and password
username="uceedoh"
password="wekgaQ-3jinmu-forrow"

# Command to kill all screen sessions
kill_command="screen -ls | awk '/(Attached)|(Detached)/ {print substr(\\\$1, 1, length(\\\$1)-1)}' | xargs -I {} screen -S {} -X quit"

# Loop through each server in the list and connect via SSH
for server in "${servers[@]}"; do
    # Use expect to automate SSH login with password
    if ! expect -c "
        set timeout -1
        spawn ssh $username@$server
        expect {
            \"yes/no\" { send \"yes\r\"; exp_continue }
            \"assword:\" { send \"$password\r\" }
        }
        expect \"\$ \"
        send \"$kill_command\r\"
        expect \"\$ \"
        send \"exit\r\"
        expect \"\$ \"
    "; then
        echo "Failed to connect to $server"
    else
        echo "Successfully killed screen sessions on $server"
    fi
done