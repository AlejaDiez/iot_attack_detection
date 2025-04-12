/** Expresión regular para obtener los comentarios */
const COMMENT_REGEX = /\/\*[\s\S]*?\*\/|\/\/.*/g;
/** Expresión regular para obtener los nombres de las funciones */
const FUNCTION_REGEX =
    /(?:function|const|let|var)\s+((intcp(_router|_iot|_computer)?|cmd_[A-Za-z0-9$_]+|atk_[A-Za-z0-9$_]+))\s*=?\s*\([A-Za-z0-9$_,.\s]*\)\s*(?:=>)?\s*{/g;

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
