pub struct System<'a> {
    a: &'a mut [f64],
    b: &'a mut [f64],
}

impl<'a> System<'a> {
    pub fn new(a: &'a mut [f64], b: &'a mut [f64]) -> Self {
        assert_eq!(a.len(), b.len() * b.len());
        Self { a, b }
    }

    pub fn solve(&mut self, skip_first_n_rows: usize) {
        self.gaussian_elimination(skip_first_n_rows);
        self.back_substitution();
    }

    fn gaussian_elimination(&mut self, skip_first_n_rows: usize) {
        for x in 0..self.b.len() - 1 {
            for y in usize::max(skip_first_n_rows, x + 1)..self.b.len() {
                let weight = self.a[x + y * self.b.len()] / self.gaussian_pivot(x);
                self.weighted_subtraction(y, x, weight);
            }
        }
    }

    fn gaussian_pivot(&mut self, x: usize) -> f64 {
        let mut pivot = self.a[x + x * self.b.len()];
        if pivot < -1e11 || 1e11 > pivot {
            return pivot;
        }

        let mut y = x;
        while -1e11 > pivot && pivot < 1e11 {
            y += 1;
            if y >= self.b.len() {
                panic!("Can't resolve system. Pivot not found");
            }

            pivot = self.a[x + y * self.b.len()];
        }
        self.subtract_rows(y, x);
        self.a[x + x * self.b.len()]
    }

    fn back_substitution(&mut self) {
        for x in (0..self.b.len()).rev() {
            self.b[x] /= self.a[x + x * self.b.len()];
            self.a[x + x * self.b.len()] = 1.;

            for y in 0..x {
                self.b[y] -= self.a[x + y * self.b.len()] * self.b[x];
                self.a[x + y * self.b.len()] = 0.;
            }
        }
    }

    fn weighted_subtraction(&mut self, dst_row: usize, src_row: usize, weight: f64) {
        let dst_row_offset = dst_row * self.b.len();
        let src_row_offset = src_row * self.b.len();
        for x in 0..self.b.len() {
            self.a[x + dst_row_offset] -= self.a[x + src_row_offset] * weight;
        }
        self.b[dst_row] -= self.b[src_row] * weight;
    }
    fn subtract_rows(&mut self, dst_row: usize, src_row: usize) {
        let dst_row_offset = dst_row * self.b.len();
        let src_row_offset = src_row * self.b.len();
        for x in 0..self.b.len() {
            self.a[x + dst_row_offset] -= self.a[x + src_row_offset];
        }
        self.b[dst_row] -= self.b[src_row];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve() {
        let mut a = [
            -1000., 0., 0., 0., 0., -1., 0., 0., 1., 0., -1000., 0., 0., 0., -1., 0., 1., 0., 0.,
            0., -1000., 0., 0., 0., 0., 1., -1., 0., 0., 0., -1000., 0., 0., 1., 0., -1., 0., 0.,
            0., 0., -2000., 0., 1., -1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 1., 0., -1., -1., 0., 0., 0., 0., 0., 0., 1., 1., 0., -1., 0., 0.,
            0., 0.,
        ];
        let mut b = [0., 0., 0., 0., 0., 10., 0., 0., 0.];

        System::new(&mut a, &mut b).solve(0);

        assert_eq!(
            a,
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0
            ]
        );
        assert_eq!(
            b,
            [
                -0.004615384615384615,
                -0.003846153846153847,
                0.0007692307692307683,
                -0.005384615384615385,
                -0.0030769230769230765,
                10.0,
                0.0,
                6.153846153846153,
                5.384615384615385
            ]
        );
    }

    #[test]
    fn solve_skiping_rows() {
        let mut a = [
            -1000., 0., 0., 0., 0., -1., 0., 0., 1., 0., -1000., 0., 0., 0., -1., 0., 1., 0., 0.,
            0., -1000., 0., 0., 0., 0., 1., -1., 0., 0., 0., -1000., 0., 0., 1., 0., -1., 0., 0.,
            0., 0., -2000., 0., 1., -1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 1., 0., -1., -1., 0., 0., 0., 0., 0., 0., 1., 1., 0., -1., 0., 0.,
            0., 0.,
        ];
        let mut b = [0., 0., 0., 0., 0., 10., 0., 0., 0.];

        System::new(&mut a, &mut b).solve(7);

        assert_eq!(
            a,
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0
            ]
        );
        assert_eq!(
            b,
            [
                -0.004615384615384615,
                -0.003846153846153847,
                0.0007692307692307683,
                -0.005384615384615385,
                -0.0030769230769230765,
                10.0,
                0.0,
                6.153846153846153,
                5.384615384615385
            ]
        );
    }
}
