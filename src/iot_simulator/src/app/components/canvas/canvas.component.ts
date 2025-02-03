import { AfterViewInit, Component, ElementRef, ViewChild } from "@angular/core";

@Component({
    selector: "app-canvas",
    imports: [],
    templateUrl: "canvas.component.html",
    styleUrl: "canvas.component.css",
})
export class CanvasComponent implements AfterViewInit {
    @ViewChild("scrollable")
    scrollable!: ElementRef<HTMLDivElement>;

    ngAfterViewInit(): void {
        this.scrollable.nativeElement.scrollTo({
            top:
                (this.scrollable.nativeElement.scrollHeight -
                    this.scrollable.nativeElement.clientHeight) /
                2,
            left:
                (this.scrollable.nativeElement.scrollWidth -
                    this.scrollable.nativeElement.clientWidth) /
                2,
            behavior: "instant",
        });
    }
}
