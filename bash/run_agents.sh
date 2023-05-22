#!/bin/bash

# Define the list of servers to connect to
servers=("kiev.ee.ucl.ac.uk" "lyon.ee.ucl.ac.uk" "verona.ee.ucl.ac.uk" "riga.ee.ucl.ac.uk" "oslo.ee.ucl.ac.uk" "monaco.ee.ucl.ac.uk" "chester.ee.ucl.ac.uk" "budapest.ee.ucl.ac.uk")

# Define the SSH username and password
username="uceedoh"
password="wekgaQ-3jinmu-forrow"
sessionname="320_nsfnet_sweep_session"

# Define the command to run in each screen session
command_to_run="cd git/vone_drl ; conda activate vone-drl ; wandb agent micdoh/VONE-DRL/2xf21vno"

# Loop through each server in the list and connect via SSH
for server in "${servers[@]}"; do
    # Use expect to automate SSH login with password
    if ! expect -c "
        set timeout 30
        spawn ssh $username@$server
        expect {
            \"yes/no\" { send \"yes\r\"; exp_continue }
            \"assword:\" { send \"$password\r\" }
        }
        expect {
            \"$ \"
            {
                send \"screen -dmS $sessionname\r\"
                expect \"$\"
                send \"screen -S $sessionname -p 0 -X stuff '$command_to_run\n'\"
                send \"exit\r\"
                expect {
                    \"$\"
                    {
                        send_user \"Successfully ran command on $server\n\"
                        exit 0
                    }
                    timeout { send_user \"Timeout waiting for command to complete on $server\n\"; exit 1 }
                    eof { send_user \"Connection closed unexpectedly on $server\n\"; exit 1 }
                    \"Connection refused\" { send_user \"Connection refused on $server\n\"; exit 1 }
                    \"No route to host\" { send_user \"No route to host on $server\n\"; exit 1 }
                    \"Name or service not known\" { send_user \"Unknown host on $server\n\"; exit 1 }
                    default { send_user \"Unexpected output on $server\n\"; exit 1 }
                }
            }
            timeout { send_user \"Timeout waiting for prompt on $server\n\"; exit 1 }
            eof { send_user \"Connection closed unexpectedly on $server\n\"; exit 1 }
            \"Connection refused\" { send_user \"Connection refused on $server\n\"; exit 1 }
            \"No route to host\" { send_user \"No route to host on $server\n\"; exit 1 }
            \"Name or service not known\" { send_user \"Unknown host on $server\n\"; exit 1 }
            default { send_user \"Unexpected output on $server\n\"; exit 1 }
        }
    "; then
        echo "Failed to connect to $server"
    fi
done

