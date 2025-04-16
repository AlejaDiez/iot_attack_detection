/** Expresión regular para obtener los comentarios */
const COMMENT_REGEX = /\/\*[\s\S]*?\*\/|\/\/.*/g;
/** Expresión regular para obtener los nombres de las funciones */
const FUNCTION_REGEX =
    /(?:function|const|let|var)\s+((intcp(_router|_iot|_computer)?|cmd_[A-Za-z0-9$_]+|atk_[A-Za-z0-9$_]+))\s*=?\s*\([A-Za-z0-9$_,.\s]*\)\s*(?:=>)?\s*{/g;
/** Expresión regular para obtener los parámetros de las funciones */
const PARAMS_REGEX = /\(([^)]*)\)/;

/**
 * Función para parsear un script de JavaScript y obtener las funciones que sean interceptores, comandos o ataques.
 *
 * @param script - El script de JavaScript a parsear.
 * @return - Una función que contiene las funciones interceptores, comandos o ataques.
 */
export const parseScript = (script: string): Function => {
    // Eliminar comentarios
    script = script.replace(COMMENT_REGEX, "");

    // Obtener los nombres de las funciones que sean interceptores, comandos o ataques
    const functions = [...script.matchAll(FUNCTION_REGEX)].map((e) => e[1]);

    // Exportar las funciones
    script = script.concat("\n", "return { ", functions.join(", "), " };\n");

    // Compilar el script
    return new Function(script);
};

/**
 * Funcion que te devuelve el numero de parametros de una funcion
 *
 * @param fn - La funcion a analizar.
 * @returns - El numero de parametros de la funcion, si hay un spread devuelve infinity.
 */
export const paramsCount = (fn: Function): number => {
    const fnStr = fn.toString();
    const paramsMatch = fnStr.match(PARAMS_REGEX);

    if (!paramsMatch) return 0;

    const params = paramsMatch[1].split(",").map((e) => e.trim());
    const spread = params.find((e) => e.includes("..."));

    return spread ? Infinity : params.length;
};
