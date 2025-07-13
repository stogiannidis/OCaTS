#!/bin/bash

# Help message
print_help() {
    echo
    echo "Usage: sinteractive [OPTIONS]"
    echo
    echo "Available arguments:"
    echo "  --part    Slurm partition.            Default: main "
    echo "  --qos     Quality Of Service.         Default: normal "
    echo "  --time    Running time limit.         Default: 0-2:00:00 "
    echo "  --cpu     Number of cpu cores.        Default: 4 "
    echo "  --mem     Amount of RAM memory (GB).  Default: 24GB "
    echo "  --gpu     Number of GPU cards.        Default: 0 "
    echo

    exit 1
}

if [[ "$#" -eq 0 ]]; then
    echo ""
    echo "No arguments provided. Using defaults."
    echo "Run 'sinteractive --help' to check for available options."
fi

# Default values
PART=main
QOS=normal
TIME="2:00:00"
CPU=4
MEM="24G"
GPU=rtx_4090:1

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --part)
            PART=$2
            shift 2
            ;;
        --qos)
            QOS=$2
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpu)
            CPU=$2
            shift 2
            ;;
        --mem)
            MEM="$2G"
            shift 2
            ;;
        --gpu)
            GPU=$2
            shift 2
            ;;
        --help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

# Output the configuration
echo ""
echo "Configuration:"
echo "  Partition:        $PART"
echo "  QoS:              $QOS"
echo "  Time Limit:       $TIME"
echo "  CPU Cores:        $CPU"
echo "  RAM Memory:       $MEM"
echo "  GPU Cards:        $GPU"
echo ""

########################################
#
# TRAP SIGINT AND SIGTERM OF THIS SCRIPT
function control_c {
    echo -en "\n SIGINT: TERMINATING SLURM JOBID $JOBID AND EXITING \n"
    scancel $JOBID
    rm interactive.sbatch
    exit $?
}
trap control_c SIGINT
trap control_c SIGTERM
#
# SBATCH FILE FOR ALLOCATING COMPUTE RESOURCES TO RUN NOTEBOOK SERVER

create_sbatch() {
cat << EOF
#!/bin/bash
#
#SBATCH --job-name interactive
#SBATCH --partition $PART
#SBATCH --qos $QOS
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --time=$TIME
#SBATCH --gpus=$GPU
#SBATCH --mem=$MEM
##SBATCH -J interactive
#SBATCH -o $CWD/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

#
echo `date`
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  \$launch "
#module load cuda/11.1
while true; do
        eval \$launch
done
EOF
}
#
## CREATE SESSION LOG FOLDER 
#if [ ! -d session_logs ] ; then
#   mkdir session_logs
#fi
##
# CREATE INTERACTIVE SBATCH FILE
export CWD=`pwd`

create_sbatch > interactive.sbatch

#
# START NOTEBOOK SERVER
#
export JOBID=$(sbatch interactive.sbatch  | awk '{print $4}')

if [ -z "${JOBID}" ]; then
  # echo  "ERROR"
  exit 0
fi

NODE=$(squeue -hj $JOBID -O nodelist )
if [[ -z "${NODE// }" ]]; then
   echo  " "
   echo -n "    WAITING FOR RESOURCES TO BECOME AVAILABLE (CTRL-C TO EXIT) ..."
fi
while [[ -z "${NODE// }" ]]; do
   echo -n "."
   sleep 3
   NODE=$(squeue -hj $JOBID -O nodelist )
done
  HOST_NAME=$(squeue -j $JOBID -h -o  %B)
#  HOST_IP=$(ssh -q $HOST_NAME 'hostname -i')
  HOST_IP=$(grep -i $HOST_NAME /etc/hosts | awk '{ print $1 }')
  TIMELIM=$(squeue -hj $JOBID -O timeleft )
  if [[ $TIMELIM == *"-"* ]]; then
  DAYS=$(echo $TIMELIM | awk -F '-' '{print $1}')
  HOURS=$(echo $TIMELIM | awk -F '-' '{print $2}' | awk -F ':' '{print $1}')
  MINS=$(echo $TIMELIM | awk -F ':' '{print $2}')
  TIMELEFT="THIS SESSION WILL TIMEOUT IN $DAYS DAY $HOURS HOUR(S) AND $MINS MINS "
  else
  HOURS=$(echo $TIMELIM | awk -F ':' '{print $1}' )
  MINS=$(echo $TIMELIM | awk -F ':' '{print $2}')
  TIMELEFT="THIS SESSION WILL TIMEOUT IN $HOURS HOUR(S) AND $MINS MINS "
  fi
  echo " "
  echo " "
  echo "  --------------------------------------------------------------------"
  echo "    INTERACTIVE SESSION STARTED ON NODE $NODE           "
  echo "    $TIMELEFT"
  echo "    SESSION LOG WILL BE STORED IN interactive_${JOBID}.out  "
  echo "  --------------------------------------------------------------------"
  echo "  "
  echo "    TO ACCESS THIS COMPUTE NODE, USE THIS IN YOUR IDE: "
  echo "  "
  echo "    $USER@${HOST_IP}  "
  echo "  "
  echo "  --------------------------------------------------------------------"
  echo "  "
  echo "    TO TERMINATE THIS SESSION ISSUE THE FOLLOWING COMMAND: "
  echo "  "
  echo "       scancel $JOBID "
  echo "  "
  echo "  --------------------------------------------------------------------"
  echo "  "
#
# CLEANUP
  rm interactive.sbatch
#
# EOF
