use knut::einsum;
use knut::NDArray;

#[test]
fn transpose() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);

    let result: NDArray<i32> = einsum("ij->ji", &vec![x]).unwrap();

    assert_eq!(result.data, vec![0, 3, 1, 4, 2, 5]);
    assert_eq!(result.shape, vec![3, 2]);
}

#[test]
fn sum() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);

    let result: NDArray<i32> = einsum("ij->", &vec![x]).unwrap();

    assert_eq!(result.data, vec![15]);
    assert_eq!(result.shape, vec![1]);
}

#[test]
fn column_sum() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);

    let result: NDArray<i32> = einsum("ij->j", &vec![x]).unwrap();

    assert_eq!(result.data, vec![3, 5, 7]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn row_sum() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);

    let result: NDArray<i32> = einsum("ij->i", &vec![x]).unwrap();

    assert_eq!(result.data, vec![3, 12]);
    assert_eq!(result.shape, vec![2]);
}

#[test]
fn matrix_vector_multiplication() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);
    let y = NDArray::new((0..3).collect(), "i32", vec![3]);

    let result: NDArray<i32> = einsum("ik,k->i", &vec![x, y]).unwrap();

    assert_eq!(result.data, vec![5, 14]);
    assert_eq!(result.shape, vec![2]);
}

#[test]
fn matrix_matrix_multiplication() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);
    let y = NDArray::new((0..15).collect(), "i32", vec![3, 5]);

    let result: NDArray<i32> = einsum("ik,kj->ij", &vec![x, y]).unwrap();

    assert_eq!(result.data, vec![25, 28, 31, 34, 37, 70, 82, 94, 106, 118]);
    assert_eq!(result.shape, vec![2, 5]);
}

#[test]
fn vector_dot_product() {
    let x = NDArray::new((0..3).collect(), "i32", vec![3]);
    let y = NDArray::new((3..6).collect(), "i32", vec![3]);

    let result: NDArray<i32> = einsum("i,i->", &vec![x, y]).unwrap();

    assert_eq!(result.data, vec![14]);
    assert_eq!(result.shape, vec![1]);
}

#[test]
fn matrix_dot_product() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);
    let y = NDArray::new((6..12).collect(), "i32", vec![2, 3]);

    let result: NDArray<i32> = einsum("ij,ij->", &vec![x, y]).unwrap();

    assert_eq!(result.data, vec![145]);
    assert_eq!(result.shape, vec![1]);
}

#[test]
fn hadamard_product() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);
    let y = NDArray::new((6..12).collect(), "i32", vec![2, 3]);

    let result: NDArray<i32> = einsum("ij,ij->ij", &vec![x, y]).unwrap();

    assert_eq!(result.data, vec![0, 7, 16, 27, 40, 55]);
    assert_eq!(result.shape, vec![2, 3]);
}

#[test]
fn outer_product() {
    let x = NDArray::new((0..3).collect(), "i32", vec![3]);
    let y = NDArray::new((3..7).collect(), "i32", vec![4]);

    let result: NDArray<i32> = einsum("i,j->ij", &vec![x, y]).unwrap();

    assert_eq!(result.data, vec![0, 0, 0, 0, 3, 4, 5, 6, 6, 8, 10, 12]);
    assert_eq!(result.shape, vec![3, 4]);
}

#[test]
fn batch_matrix_multiplication() {
    let x = NDArray::new((0..30).collect(), "i32", vec![3, 2, 5]);
    let y = NDArray::new((0..45).collect(), "i32", vec![3, 5, 3]);

    let result: NDArray<i32> = einsum("ijk,ikl->ijl", &vec![x, y]).unwrap();

    assert_eq!(
        result.data,
        vec![
            90, 100, 110, 240, 275, 310, 1290, 1350, 1410, 1815, 1900, 1985, 3990, 4100, 4210,
            4890, 5025, 5160
        ]
    );
    assert_eq!(result.shape, vec![3, 2, 3]);
}

#[test]
fn tensor_contraction() {
    let x = NDArray::new((0..210).collect(), "i32", vec![2, 3, 5, 7]);
    let y = NDArray::new((0..360).collect(), "i32", vec![1, 4, 3, 6, 5]);

    let result: NDArray<i32> = einsum("pqrs,tuqvr->pstuv", &vec![x, y]).unwrap();

    assert_eq!(result.shape, vec![2, 7, 1, 4, 6]);
}

#[test]
fn bilinear_transformation() {
    let x = NDArray::new((0..6).collect(), "i32", vec![2, 3]);
    let y = NDArray::new((0..105).collect(), "i32", vec![5, 3, 7]);
    let z = NDArray::new((0..14).collect(), "i32", vec![2, 7]);

    let result: NDArray<i32> = einsum("ik,jkl,il->ij", &vec![x, y, z]).unwrap();

    assert_eq!(
        result.data,
        vec![1008, 2331, 3654, 4977, 6300, 9716, 27356, 44996, 62636, 80276]
    );
    assert_eq!(result.shape, vec![2, 5]);
}
