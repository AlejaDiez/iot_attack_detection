import { Pipe, PipeTransform } from "@angular/core";

@Pipe({
    name: "name",
})
export class NamePipe implements PipeTransform {
    public transform(value: string): string {
        return value.replaceAll("_", " ").trim();
    }
}
