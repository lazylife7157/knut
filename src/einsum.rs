use crate::ndarray::NDArray;
use regex::Regex;
use std::collections::HashMap;

use weld::data::WeldVec;
use weld::{Data, WeldConf, WeldContext, WeldError, WeldModule, WeldValue};

lazy_static! {
    static ref RE_EINSUM: Regex = Regex::new(r"[a-z](,[a-z])*->[a-z]*").unwrap();
    static ref WELD_CONFIG: WeldConf = WeldConf::new();
}

fn compile_and_run<T>(
    context: &mut WeldContext,
    code: &str,
    input: &Vec<WeldVec<T>>,
) -> Result<WeldVec<T>, WeldError>
where
    T: std::clone::Clone,
{
    let module = WeldModule::compile(code, &WELD_CONFIG)?;
    let input = WeldValue::new_from_data(input.as_ptr() as Data);
    unsafe {
        let result = module.run(context, &input)?;
        let data = result.data() as *const WeldVec<T>;
        let data = (*data).clone();
        Ok(data)
    }
}

pub fn compile_equation<T>(equation: &str, operands: &Vec<NDArray<T>>) -> (String, Vec<usize>) {
    assert!(RE_EINSUM.is_match(equation));

    let dtype = operands[0].dtype;
    let mut dims: HashMap<char, usize> = HashMap::new();

    let terms: Vec<&str> = equation.trim().split("->").collect();
    assert_eq!(terms.len(), 2);
    let args: Vec<&str> = terms[0].split(",").collect();
    assert_eq!(args.len(), operands.len());
    let dest = terms[1];

    let argname = |i: usize| format!("arg{}", i);

    for (i, arg) in args.iter().enumerate() {
        for (j, c) in arg.chars().enumerate() {
            dims.insert(c, operands[i].shape[j]);
        }
    }

    let mut output_shape = dest.chars().map(|i| dims[&i]).collect();

    let mut summation_indices: Vec<char> = args
        .join("")
        .chars()
        .filter(|&c| !dest.contains(c))
        .collect();
    summation_indices.sort();
    summation_indices.dedup();

    let index = |axes: &str| -> String {
        let _axes: Vec<char> = axes.chars().collect();
        let num_axes = axes.len();
        let mut index = Vec::new();
        for (i, axis) in _axes.iter().enumerate() {
            let temp = (i + 1..num_axes)
                .map(|j| dims[&_axes[j]])
                .product::<usize>();
            let temp = format!("({} * {}L)", axis, temp);
            index.push(temp);
        }

        index.join(" + ")
    };

    let mut code = args
        .iter()
        .enumerate()
        .map(|(i, arg)| format!("lookup({}, {})", argname(i), index(arg)))
        .collect::<Vec<_>>()
        .join(" * ");

    for i in summation_indices.iter().rev() {
        code = format!(
            "result(for(rangeiter(0L, {}L, 1L), merger[{}, +], |b, {}, t| merge(b, {})))",
            dims[i], dtype, i, code
        );
    }

    let mut _dtype = String::from(dtype);
    for i in dest.chars().rev() {
        code = format!(
            "result(for(rangeiter(0L, {}L, 1L), appender[{}], |b, {}, t| merge(b, {})))",
            dims[&i], _dtype, i, code
        );
        _dtype = format!("vec[{}]", _dtype);
    }

    for _ in 1..dest.len() {
        code = format!("flatten({})", code);
    }

    if dest.len() == 0 {
        code = format!("merge(appender[{}], {})", dtype, code);
        output_shape = vec![1];
    }

    let params = (0..args.len())
        .map(|i| format!("{}: vec[{}]", argname(i), dtype))
        .collect::<Vec<String>>()
        .join(", ");
    code = format!("|{}| {}", params, code);

    (code, output_shape)
}

pub fn einsum<T>(equation: &str, operands: &Vec<NDArray<T>>) -> Result<NDArray<T>, WeldError>
where
    T: std::clone::Clone,
{
    let mut context = WeldContext::new(&WELD_CONFIG)?;
    let (code, output_shape) = compile_equation(equation, operands);
    let input: Vec<_> = operands.iter().map(|o| WeldVec::from(&o.data)).collect();
    let result = compile_and_run(&mut context, &code, &input).unwrap();
    let data = (0..result.len)
        .map(|i| unsafe { (*result.data.offset(i as isize)).clone() })
        .collect();

    Ok(NDArray::new(data, operands[0].dtype, output_shape))
}
