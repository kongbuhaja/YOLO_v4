#!/bin/bash

# Define color codes
RED='\033[0;91m'
GREEN='\033[0;92m'
BLUE='\033[0;94m'
YELLOW='\033[0;93m'
NC='\033[0m' # No Color

# Function to print a line
print_line() {
    printf "${BLUE}"
    printf '%*s\n' "$full_length" '' | tr ' ' =
    printf "${NC}"
}

# Function to get detailed CPU information
get_cpu_info() {
    # Getting CPU information
    local cpu_model=$(lscpu | grep "Model name" | sed -e 's/Model name:[ \t]*//')
    local cpu_count=$(lscpu | grep "Socket(s):" | awk '{print $2}')
    local cpu_cores=$(lscpu | grep "^CPU(s):" | awk '{print $2}')

    echo "$cpu_model|$cpu_count|$cpu_cores"
}

get_cpu_usage() {
    local -a cpu_usages

    while read -r core usage; do
        cpu_usages["$core"]=$usage
    done < <(mpstat -P ALL 1 1 | awk '/^Average:/ && $2 ~ /[0-9]+/ {print $2, $3}')

    echo "${cpu_usages[@]}"
}


get_cpu_temp() {
    local cpu_temp=$(sensors | awk '/^Package id 0:/ {print $4}' | sed 's/+//;s/°C//')
    
    echo "${cpu_temp:0:-2}"
}

print_cpu_info() {
    local cpu_model=$1 
    local cpu_count=$2 
    local cpu_cores=$3
    local cpu_info=$4
    local cpu_temp=$5
    cpu_usages=$6
    local length=$((($full_length-1)/8-4))

    # title
    if [ "$cpu_temp" -ge 70 ]; then
        local temp_color=${RED}
    elif [ "$cpu_temp" -ge 60 ]; then
        local temp_color=${YELLOW}
    else
        local temp_color=${GREEN}
    fi

    local do="°"
    local fuck=${#do}
    local text="${GREEN}‖ $cpu_count CPUs ‖ $cpu_cores cores ‖ temp: ${temp_color}$cpu_temp°C ${GREEN}‖"
    local t_length=$((${#text}-29-fuck))
    local l_pad=$(((full_length-t_length)/2))
    local r_pad=$((full_length-l_pad-t_length))
    printf "${BLUE}%*s" "$l_pad" '' | tr ' ' =
    printf "$text"
    printf "${BLUE}%*s\n" "$r_pad" '' | tr ' ' = 

    # cpu model
    local t_length=${#cpu_model}
    local l_pad=$(((full_length-t_length-2)/2))
    local r_pad=$((full_length-l_pad-t_length-2))
    printf "‖${GREEN}%*s$cpu_model%*s${BLUE}‖\n" "$l_pad" '' "$r_pad"
    
    # cpu usage
    for core in "${!cpu_usages[@]}"; do
        if [ $(("$core"%8)) -eq 0 ]; then
            printf "${BLUE}‖"
        fi
        if [ "${cpu_usages[$core]:0:-3}" -ge 70 ]; then
            local usage_color=${RED}
        elif [ "${cpu_usages[$core]:0:-3}" -ge 60 ]; then
            local usage_color=${YELLOW}
        else
            local usage_color=${NC}
        fi
        local i_length=$((2-${#core}))
        local u_length=$(($length-${#cpu_usages[$core]}))
        local ru_pad=$((u_length/2))
        local lu_pad=$((u_length-ru_pad))
        
        printf "${NC}${GREEN}%*s%s${NC} ${usage_color}%*s%s%%%*s${BLUE}‖" "$i_length" '' "$core" "$lu_pad" '' "${cpu_usages[$core]:0:-1}" "$ru_pad" ''
        if [ $(("$core"%8)) -eq 7 ]; then
            printf "\n"
        fi
    done
}

# Function to get GPU information
get_gpu_info() {
    # nvidia-smi 명령어를 이용하여 GPU 정보를 추출
    local gpu_info=$(nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    echo "$gpu_info"
}

print_gpu_info() {
    local gpu_info=$1

    # Calculate the number of GPUs
    local gpu_count=$(echo "$gpu_info" | grep -c '^')

    # Print header for GPUs
    local length=$((full_length-48))
    local power_length=$((length/2-2))
    local memory_length=$((length-power_length))

    local text="‖ ${gpu_count} GPUs ‖"
    local t_length=${#text}
    local r_pad=$(((full_length-t_length-2)/2))
    local l_pad=$((full_length-r_pad-t_length-2))

    printf "${BLUE}‖%*s" "$l_pad" '' | tr ' ' =
    printf "${GREEN}$text"
    printf "${BLUE}%*s‖\n" "$r_pad" '' | tr ' ' = 

    # Head
    printf "${BLUE}‖%*s${GREEN}" "2"
    # Device
    local text="Device"
    local t_length=${#text}
    local r_pad=$(((25-t_length)/2))
    local l_pad=$((25-r_pad-t_length))
    printf " %*s$text" "$l_pad"
    printf "%*s" "$r_pad" 

    # Temp
    local text="Temp"
    local t_length=${#text}
    local r_pad=$(((6-t_length)/2))
    local l_pad=$((6-r_pad-t_length))
    printf "|%*s$text" "$l_pad"
    printf "%*s" "$r_pad" 

    # Power
    local text="Power"
    local t_length=${#text}
    local r_pad=$(((power_length-t_length)/2))
    local l_pad=$((power_length-r_pad-t_length))
    printf "|%*s$text" "$l_pad"
    printf "%*s" "$r_pad" 

    # Memory
    local text="Memory"
    local t_length=${#text}
    local r_pad=$(((memory_length-t_length)/2))
    local l_pad=$((memory_length-r_pad-t_length))
    printf "|%*s$text" "$l_pad"
    printf "%*s" "$r_pad" 

    # Volt
    local text="Volt"
    local t_length=${#text}
    local r_pad=$(((8-t_length)/2))
    local l_pad=$((8-r_pad-t_length))
    printf "|%*s$text" "$l_pad"
    printf "%*s" "$r_pad" 

    printf "${BLUE}‖\n"

    # GPUs info
    local p=$((power_length/2))
    local m=$((memory_length/2))
    echo "$gpu_info" | while IFS=',' read -r index name temperature power_draw power_limit memory_used memory_total utilization
    do
        if [ "${temperature%.*}" -ge 65 ]; then
            local temp_color=$RED
        elif [ "${temperature%.*}" -ge 50 ]; then
            local temp_color=$YELLOW
        else
            local temp_color=$NC
        fi
        
        # Print GPU information line with colors

        local d=$(((25-${#name})/2))
        printf "${BLUE}‖${GREEN}%2s${NC} %"$((25-d))"s%"$d"s|${temp_color}%7s${NC}|%"$p"s%-"$((power_length-p))"s|%"$m"s%-"$((memory_length-m))"s|%8s${BLUE}‖\n" \
         "$index" "$name" '' "${temperature:1}°C " "${power_draw:1:-3}/" "${power_limit:1:-3}W" "${memory_used:1}/" "${memory_total:1}Mb" "${utilization:1}%  "
    done
    printf "${NC}"
}

# Function to get Memory information
get_memory_info() {
    local mem_total=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
    local mem_free=$(awk '/MemFree:/ {print $2}' /proc/meminfo)
    local mem_buffers=$(awk '/Buffers:/ {print $2}' /proc/meminfo)
    local mem_cached=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
    local mem_used=$((mem_total-mem_free-mem_buffers-mem_cached))

    # Get total and used swap
    local swap_total=$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)
    local swap_free=$(awk '/SwapFree:/ {print $2}' /proc/meminfo)
    local swap_used=$((swap_total-swap_free))
    
    local mem_used=$(echo "scale=1; $mem_used / $mb" | bc)
    local mem_total=$(echo "scale=1; $mem_total / $mb" | bc)
    local swap_used=$(echo "scale=1; $swap_used / $mb" | bc)
    local swap_total=$(echo "scale=1; $swap_total / $mb" | bc)

    echo "$mem_used|$mem_total|$swap_used|$swap_total"
}

print_memory_info() {
    local mem_used=$1
    local mem_total=$2
    local swap_used=$3 
    local swap_total=$4
    local half_length=$(((full_length-3)/2))

    ml_pad=$(((half_length-3)/2))
    mr_pad=$((half_length-ml_pad-3))
    sl_pad=$(((half_length-4)/2))
    sr_pad=$((half_length-sl_pad-4))
    
    local text="‖ Memory ‖"
    local t_length=${#text}
    local r_pad=$(((full_length-t_length-2)/2))
    local l_pad=$((full_length-r_pad-t_length-2))

    printf "${BLUE}‖%*s" "$l_pad" '' | tr ' ' =
    printf "${GREEN}$text"
    printf "${BLUE}%*s‖\n" "$r_pad" '' | tr ' ' = 

    printf "${BLUE}‖${GREEN}%"$ml_pad"sRam%"$mr_pad"s|%"$sl_pad"sSwap%"$sr_pad"s${BLUE}‖\n"
    m=$((half_length/2))
    s=$((half_length/2))
    printf "${BLUE}‖${NC}%"$((m-${#mem_used}))"s"$mem_used"/"$mem_total"Gb%-$((m-${#mem_total}-2))s|"
    printf "%"$((s-${#swap_used}-1))"s"$swap_used"/"$swap_total"Gb%-$((s-${#swap_total}-1))s${BLUE}‖\n"

}

# Main function to display all information
print_system_status() {
    mb=$((1024*1024))

    length=10
    full_length=$((length*8+1))

    clear
    IFS='|' read -r cpu_model cpu_count cpu_cores <<< "$(get_cpu_info)"

    while true; do
        local cpu_temp=$(get_cpu_temp)
        local cpu_usages=($(get_cpu_usage))
        local gpu_info=$(get_gpu_info)
        IFS='|' read -r mem_used mem_total swap_used swap_total <<< "$(get_memory_info)"
        tput cup 0 0
        print_cpu_info "$cpu_model" "$cpu_count" "$cpu_cores" "$cpu_info" "$cpu_temp" "${cpu_usages[@]}"
        print_gpu_info "$gpu_info"
        print_memory_info "$mem_used" "$mem_total" "$swap_used" "$swap_total"
        print_line
    done
}


print_system_status