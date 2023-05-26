

std::array<int, 6> l = {1, 2, 4, 5, 3, 3};
int s = 6;

std::array<int, 6> r;


for(int i_idx = 0; i_idx < l_len; i_idx++) {
    for(int j_idx = i_idx; j_idx < l_len; j_idx++) {
        
        if (l[i_idx] + l[j_idx] == s) {
            r.push_back(s)
        }

    }

}