#include <mpi.h>
#include <algorithm>
#include <vector>
#include <boost/sort/spreadsort/float_sort.hpp>
// #include <parallel/algorithm>


void merge(bool small, int partner_size,
            std::vector<float> &local,
            std::vector<float> &partner,
            std::vector<float> &merged) {
    int n = local.size();
    if (small) {
        int i=0, j=0;
        for (int k=0; k<n; k++) {
            if (j>=partner_size || (i<n && local[i]<partner[j])) {
                merged[k] = local[i++];
            } else {
                merged[k] = partner[j++];
            }
        }
    } else {
        int i=n-1, j=partner_size-1;
        for (int k=n-1; k>=0; k--) {
            if (j<0 || (i>=0 && local[i]>partner[j])) {
                merged[k] = local[i--];
            } else {
                merged[k] = partner[j--];
            }
        }
    }
}


bool exchange_with_partner(int rank, int p_rank, 
                            int partner_size,
                            std::vector<float> &local,
                            std::vector<float> &partner,
                            std::vector<float> &merged,
                            MPI_Comm comm, int tag) {
    if (p_rank==-1 || local.empty() || partner_size==0) return false;

    MPI_Status status;
    bool need_exchange;


    float send_val, recv_val;
    if (rank < p_rank) send_val = local.back();
    else send_val = local.front();

    MPI_Sendrecv(&send_val, 1, MPI_FLOAT, p_rank, tag,
                &recv_val, 1, MPI_FLOAT, p_rank, tag,
                comm, &status);

    if (rank < p_rank) need_exchange = send_val > recv_val;
    else need_exchange = send_val < recv_val;


    if (!need_exchange) return false;

    MPI_Sendrecv(local.data(), local.size(), MPI_FLOAT, p_rank, tag+100,
                partner.data(), partner_size, MPI_FLOAT, p_rank, tag+100, 
                comm, &status);

    if (rank < p_rank)
        merge(true, partner_size, local, partner, merged);
    else 
        merge(false, partner_size, local, partner, merged);

    std::swap(local, merged);
    return true;
}


int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
   
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num = atoi(argv[1]);
    const char *const input_filename = argv[2];
    const char *const output_filename = argv[3];

    MPI_File input_file, output_file;
    
    int base = num / size;
    int remainder = num % size;

    int local_num = base + (rank < remainder ? 1 : 0);
    MPI_Offset elem_off = (MPI_Offset)base * rank + std::min(rank, remainder);
    MPI_Offset byte_off = elem_off * (MPI_Offset)sizeof(float);


    int odd_partner, even_partner;
    int odd_partner_size, even_partner_size;
    
    if (rank & 1) {
        odd_partner = rank - 1;
        even_partner = rank + 1;
    } else {
        odd_partner = rank + 1;
        even_partner = rank - 1;
    }

    if (odd_partner < remainder) {
        odd_partner_size = base+1;
    } else {
        odd_partner_size = base;
    }
    if (even_partner < remainder) {
        even_partner_size = base+1;
    } else {
        even_partner_size = base;
    }

    if (odd_partner<0 || odd_partner>=size) odd_partner = -1;
    if (even_partner<0 || even_partner>=size) even_partner = -1;



    std::vector<float> local_partition(local_num);
    // std::vector<float> local_partition;
    // local_partition.resize(local_num);

    std::vector<float> partner_partition(base + 1);
    // std::vector<float> partner_partition;
    // partner_partition.resize(base + 1);
    
    std::vector<float> merged_partition(local_num);
    // std::vector<float> merged_partition;
    // merged_partition.resize(local_num);


    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, byte_off, local_partition.data(), 
                            local_num, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // std::sort(local_partition.begin(), local_partition.end());
    boost::sort::spreadsort::float_sort(local_partition.begin(), local_partition.end());
    // __gnu_parallel::sort(local_partition.begin(), local_partition.end());

    for (int step=0; step < size+1; step+=2) {
        bool exchanged = false;
        int base_tag = 1000 + step * 10;

        // ---- Even phase ----
        if (even_partner != -1) {
            exchanged |= exchange_with_partner(rank, even_partner, even_partner_size,
                                    local_partition, partner_partition, merged_partition,
                                    MPI_COMM_WORLD, base_tag+10);
        }

        // ---- Odd phase ----
        if (odd_partner != -1) {
            exchanged |= exchange_with_partner(rank, odd_partner, odd_partner_size,
                                    local_partition, partner_partition, merged_partition,
                                    MPI_COMM_WORLD, base_tag+20);
        }
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, byte_off, local_partition.data(), 
                            local_num, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    MPI_Finalize();
    return 0;
}
